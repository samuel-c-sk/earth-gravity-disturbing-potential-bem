#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ================== Parallelization switch ==================
#ifdef USE_OPENMP
  #include <omp.h>
  #define OMP_PARALLEL_FOR _Pragma("omp parallel for")
  #define OMP_PARALLEL_FOR_REDUCTION(x) _Pragma(#x)
  #define OMP_SET_NUM_THREADS(n) omp_set_num_threads(n)
  #define OMP_GET_MAX_THREADS() omp_get_max_threads()
#else
  #define OMP_PARALLEL_FOR
  #define OMP_PARALLEL_FOR_REDUCTION(x)
  #define OMP_SET_NUM_THREADS(n)
  #define OMP_GET_MAX_THREADS() 1
#endif
// ============================================================

#define R 6378000.0
#define GM 3.986004418e14
#define D 500

#define PI 3.14159265358979323846

#define TOLERANCE 1e-8
#define MAX_ITER 10000

constexpr int MAX_TRI = 6;   // maximum triangles (max neighbors supported)
constexpr int NGAUSS = 7;    // number of Gauss points per triangle
const double EPS_DEN = 1e-30;     // to avoid division by zero
const double MAX_STEP = 1e12;     // maximum step size in iterations

enum class Dataset { N902, N3602 };

double compute_rmse_from_file(const char* path, int n) {
    FILE* f = fopen(path, "r");
    if (!f) { perror(path); return -1.0; }

    double sum = 0.0, v = 0.0;
    for (int i = 0; i < n; ++i) {
        if (fscanf(f, "%lf", &v) != 1) {
            fprintf(stderr, "Read error in %s at line %d\n", path, i);
            fclose(f);
            return -1.0;
        }
        sum += v * v;
    }
    fclose(f);
    return sqrt(sum / n);
}

bool write_error_file(const char* path, const double* u, int n) {
    FILE* f = fopen(path, "w");
    if (!f) { perror(path); return false; }

    for (int i = 0; i < n; ++i) {
        fprintf(f, "%.15e\n", u[i] - (GM / R));
    }
    fclose(f);
    return true;
}

// --------------------------------------------------
// Solve linear system for gravitational potential u
// --------------------------------------------------
double* bicgstab(double** A, double* b, int n) {
    double* r = new double[n];
    double* r_hat = new double[n];
    double* v = new double[n];
    double* p = new double[n];
    double* s = new double[n];
    double* t = new double[n];
    double* Ax = new double[n];
    double* x = new double[n];

    double rho_old = 1, rho_new, alpha = 1, omega = 1, beta;

OMP_PARALLEL_FOR
    for (int j = 0; j < n; j++) {
        x[j] = 0.0;
        r[j] = 0.0;
        Ax[j] = 0.0;
        r_hat[j] = 0.0;
        v[j] = 0.0;
        p[j] = 0.0;
        s[j] = 0.0;
        t[j] = 0.0;
    }

    // Compute r0 = b - Ax0, for initial guess x0
OMP_PARALLEL_FOR
    for (int i = 0; i < n; i++) {
        r[i] = b[i];
        r_hat[i] = r[i];
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        rho_new = 0;
OMP_PARALLEL_FOR_REDUCTION(omp parallel for reduction(+:rho_new))
        for (int i = 0; i < n; i++) {
            rho_new += r_hat[i] * r[i];
        }

        if (iter == 0) {
OMP_PARALLEL_FOR
            for (int i = 0; i < n; i++) {
               p[i] = r[i];
            }
        }
        else {
            if (fabs(omega) < EPS_DEN) {
                fprintf(stderr, "BiCGSTAB breakdown: omega too small (omega=%e)\n", omega);
                break;
            }
            beta = (rho_new / rho_old) * (alpha / omega);
OMP_PARALLEL_FOR
            for (int i = 0; i < n; i++) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // Compute v = A * p
OMP_PARALLEL_FOR
        for (int i = 0; i < n; i++) {
            v[i] = 0.0;
            for (int j = 0; j < n; j++) {
                v[i] += A[i][j] * p[j];
            }
        }

            double temp = 0; 
OMP_PARALLEL_FOR_REDUCTION(omp parallel for reduction(+:temp))
        for (int i = 0; i < n; i++) {
            temp += r_hat[i] * v[i];
        }
        
        if (fabs(temp) < 1e-30) {
            fprintf(stderr, "BiCGSTAB breakdown: r_hat·v too small (temp=%e)\n", temp);
            break;
        }
        alpha = rho_new / temp;
        if (alpha > MAX_STEP) {
            fprintf(stderr, "BiCGSTAB breakdown: alpha not finite\n");
            break;
        }
OMP_PARALLEL_FOR
        for (int i = 0; i < n; i++) {
            s[i] = r[i] - alpha * v[i];
        }

    // Check norm of s
        double s_norm = 0;
OMP_PARALLEL_FOR_REDUCTION(omp parallel for reduction(+:s_norm))
        for (int i = 0; i < n; i++) {
            s_norm += s[i] * s[i];
        }
        s_norm = sqrt(s_norm);

        if (s_norm < TOLERANCE) {
OMP_PARALLEL_FOR
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
            }
            break;
        }

        // Compute t = A * s
OMP_PARALLEL_FOR
        for (int i = 0; i < n; i++) {
            t[i] = 0.0;
            for (int j = 0; j < n; j++) {
                t[i] += A[i][j] * s[j];
            }
        }

        double t_dot_s = 0, t_dot_t = 0;
OMP_PARALLEL_FOR_REDUCTION(omp parallel for reduction(+:t_dot_s,t_dot_t))
        for (int i = 0; i < n; i++) {
            t_dot_s += t[i] * s[i];
            t_dot_t += t[i] * t[i];
        }

        if (fabs(t_dot_t) < EPS_DEN) {
            fprintf(stderr, "BiCGSTAB breakdown: t·t too small (t_dot_t=%e)\n", t_dot_t);
            break;
        }
        omega = t_dot_s / t_dot_t;
        if (omega > MAX_STEP) {
            fprintf(stderr, "BiCGSTAB breakdown: omega not finite\n");
        break;
}
OMP_PARALLEL_FOR
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i] + omega * s[i];
        }

OMP_PARALLEL_FOR
        for (int i = 0; i < n; i++) {
            r[i] = s[i] - omega * t[i];
        }

        double res_norm = 0;
OMP_PARALLEL_FOR_REDUCTION(omp parallel for reduction(+:res_norm))
        for (int i = 0; i < n; i++) {
            res_norm += r[i] * r[i];
        }
        res_norm = sqrt(res_norm);

        if (res_norm < TOLERANCE) break;
        printf("reziduum[%d] :%.10lf\n", iter, res_norm);

        rho_old = rho_new;
    }

    delete[] r;
    delete[] r_hat;
    delete[] v;
    delete[] p;
    delete[] s;
    delete[] t;
    delete[] Ax;

    return x; 
}

int main()
{
    int status = 0;

    // ===================== CONFIG =====================
    const bool RUN_EOC = true;
    const Dataset DATASET = Dataset::N3602;
    const bool VERBOSE = true;
    // ==================================================

    printf("MODE: DATASET = %s | RUN_EOC = %s\n",
           (DATASET == Dataset::N902 ? "902" : "3602"),
           (RUN_EOC ? "ON" : "OFF"));

    // ---- OpenMP info ----
#ifdef USE_OPENMP
    printf("OpenMP: ENABLED\n");

    const int NUM_THREADS = 8;
    OMP_SET_NUM_THREADS(NUM_THREADS);
#else
    printf("OpenMP: DISABLED (serial mode)\n");
#endif

    if (VERBOSE) {
        printf("Threads available = %d\n", OMP_GET_MAX_THREADS());
    }


    int N = 0;
    const char* BL_FILE = nullptr;
    const char* ELEM_FILE = nullptr;  
    
    double eta1[NGAUSS] = {
        1.0 / 3.0,
        0.79742699, 0.10128651, 0.10128651,
        0.05971587, 0.47014206, 0.47014206
    };
    double eta2[NGAUSS] = {
        1.0 / 3.0,
        0.10128651, 0.79742699, 0.10128651,
        0.47014206, 0.05971587, 0.47014206
    }; 
    double eta3[NGAUSS] = {
        1.0 / 3.0,
        0.10128651, 0.10128651, 0.79742699,
        0.47014206, 0.47014206, 0.05971587
    };

    double wk[NGAUSS] = {
    0.225,
    0.12593918, 0.12593918, 0.12593918,
    0.13239415, 0.13239415, 0.13239415
    };

    switch (DATASET) {
        case Dataset::N902:
            N = 902;
            BL_FILE = "BL-902.dat";
            ELEM_FILE = "elem_902.dat";
            break;
        case Dataset::N3602:
            N = 3602;
            BL_FILE = "BL-3602.dat";
            ELEM_FILE = "elem_3602.dat";
        break;
    }

    if (N <= 0) {
        fprintf(stderr, "Invalid dataset selection.\n");
        return 1;
    }

    double* B = nullptr, * L = nullptr, * H = nullptr, * q = nullptr, * nx = nullptr, * ny = nullptr;
    double * nz = nullptr, * X = nullptr, * Y = nullptr, * Z = nullptr, * tmp = nullptr;
    double*** XG = nullptr;
    double*** YG = nullptr;
    double*** ZG = nullptr;
    double* Gdiag = nullptr;
    double** F = nullptr;
    double* b = nullptr;
    double* u = nullptr;

    double total_area = 0;

    B = new double[N];
    L = new double[N];
    H = new double[N];
    q = new double[N];
    tmp = new double[N];

    FILE* file3 = nullptr;
    FILE* file2 = nullptr;

    int** E = nullptr;
    E = new int* [N];  
    for (int i = 0; i < N; i++) {
        E[i] = new int[NGAUSS];   
    }

    double** AA = nullptr;
    AA = new double* [N];   
    for (int i = 0; i < N; i++) {
        AA[i] = new double[MAX_TRI];   
    }

    nx = new double[N];
    ny = new double[N];
    nz = new double[N];
    X = new double[N];
    Y = new double[N];
    Z = new double[N];

// Read BL input data
    FILE* file = nullptr;
    file = fopen(BL_FILE, "r");
    if (!file) { perror(BL_FILE); status = 1; goto cleanup; }

    for (int i = 0; i < N; i++) {
        if (fscanf(file, "%lf %lf %lf %lf %lf", &B[i], &L[i], &H[i], &q[i], &tmp[i]) != 5) {
            fprintf(stderr, "Error reading BL file at line %d\n", i);
            status = 1; 
            goto cleanup;
        }
    }

    // Read element connectivity data
    file2 = fopen(ELEM_FILE, "r");
    if (!file2) { perror(ELEM_FILE); status = 1; goto cleanup; }

    for (int i = 0; i < N; i++) {
        if (fscanf(file2, "%d %d %d %d %d %d %d",
               &E[i][0], &E[i][1], &E[i][2], &E[i][3], &E[i][4], &E[i][5], &E[i][6]) != 7) {
            fprintf(stderr, "Error reading ELEM file at line %d\n", i);
            status = 1;
            goto cleanup;
        }
    }

OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        B[i] = B[i] * PI / 180.0;
        L[i] = L[i] * PI / 180.0;
        q[i] *= 0.00001;
    }

    // Apply EOC test setup (overwrite q and H)
    if (RUN_EOC) {
    // Override inputs for EOC/validation setup
        OMP_PARALLEL_FOR
        for (int i = 0; i < N; i++) {
            q[i] = GM / (R * R);
            H[i] = 0.0;
        }
    }

OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        nx[i] = cos(B[i]) * cos(L[i]);
        ny[i] = cos(B[i]) * sin(L[i]);
        nz[i] = sin(B[i]);
        X[i] = (R + H[i]) * nx[i];
        Y[i] = (R + H[i]) * ny[i];
        Z[i] = (R + H[i]) * nz[i];
    }

OMP_PARALLEL_FOR_REDUCTION(omp parallel for reduction(+:total_area))
    for (int i = 0; i < N; i++) {
        int neighbor_count = E[i][0];
        
        // Each surface element is triangulated using a fan around the central node.
        // At least 3 neighbors are required to form a valid triangle.
        if (neighbor_count < 3 || neighbor_count > MAX_TRI) {
            fprintf(stderr, "Invalid neighbor_count=%d at element %d\n", neighbor_count, i);
            status = 1;
            goto cleanup;
            }

        for (int j = 0; j < neighbor_count; j++) {
            int idxA = i;
            int idxB = E[i][1 + j] - 1;
            int idxC = E[i][1 + ((j + 1) % neighbor_count)] - 1;
            if (idxB < 0 || idxB >= N || idxC < 0 || idxC >= N) {
                fprintf(stderr, "Invalid neighbor index at element %d\n", i);
                status = 1;
                goto cleanup;
                }


            // vector differences
            double v1x = X[idxB] - X[idxA];
            double v1y = Y[idxB] - Y[idxA];
            double v1z = Z[idxB] - Z[idxA];

            double v2x = X[idxC] - X[idxA];
            double v2y = Y[idxC] - Y[idxA];
            double v2z = Z[idxC] - Z[idxA];

            // cross product
            double cx = v1y * v2z - v1z * v2y;
            double cy = v1z * v2x - v1x * v2z;
            double cz = v1x * v2y - v1y * v2x;

            double norm = sqrt(cx * cx + cy * cy + cz * cz);
            double area = 0.5 * norm;

            AA[i][j] = area;
            total_area += area;
        }

    }

    printf("total area:%lf\n", total_area / 3.);
    // total_area is accumulated per element; divide by 3 due to triangle counting scheme (dataset-specific)

   // --- Allocation of arrays for Gauss integration points ---

    XG = new double** [N];
    YG = new double** [N];
    ZG = new double** [N];

    for (int i = 0; i < N; i++) {
        XG[i] = new double* [MAX_TRI];
        YG[i] = new double* [MAX_TRI];
        ZG[i] = new double* [MAX_TRI];
        for (int j = 0; j < MAX_TRI; j++) {
            XG[i][j] = new double[NGAUSS];
            YG[i][j] = new double[NGAUSS];
            ZG[i][j] = new double[NGAUSS];
        }
    }

// --- Computation of global coordinates of Gauss integration points ---
OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        int neighbor_count = E[i][0];
        if (neighbor_count < 3 || neighbor_count > MAX_TRI) {
            fprintf(stderr, "Invalid neighbor_count=%d at element %d\n", neighbor_count, i);
            status = 1;
            goto cleanup;
        }

        for (int j = 0; j < neighbor_count; j++) {
            int idxA = i;
            int idxB = E[i][1 + j] - 1;
            int idxC = E[i][1 + ((j + 1) % neighbor_count)] - 1;
            if (idxB < 0 || idxB >= N || idxC < 0 || idxC >= N) {
                fprintf(stderr, "Invalid neighbor index at element %d\n", i);
                continue;
            }

            for (int k = 0; k < 7; k++) {
                XG[i][j][k] = X[idxA] * eta1[k] + X[idxB] * eta2[k] + X[idxC] * eta3[k];
                YG[i][j][k] = Y[idxA] * eta1[k] + Y[idxB] * eta2[k] + Y[idxC] * eta3[k];
                ZG[i][j][k] = Z[idxA] * eta1[k] + Z[idxB] * eta2[k] + Z[idxC] * eta3[k];
            }
        }
    }
    
// NOTE: Dense system matrix F is assembled explicitly only for small datasets
// (N <= 3602) for validation and demonstration purposes.
// Large-scale runs require matrix-free or block-based assembly on HPC clusters.

    if (N > 5000) {
        fprintf(stderr,
            "N=%d too large for explicit dense matrix assembly.\n"
            "Use matrix-free BEM formulation for large-scale runs.\n", N);
        status = 1;
        goto cleanup;
    }

    F = new double* [N];
    for (int i = 0; i < N; i++) {
        F[i] = new double[N];
    }

OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                F[i][j] = 0.0;
                continue;
            }

            double Fij = 0.0;
            int neighbor_count = E[j][0];
            if (neighbor_count < 3 || neighbor_count > MAX_TRI) {
                fprintf(stderr, "Invalid neighbor_count=%d at element %d\n", neighbor_count, j);
                status = 1;
                goto cleanup;
            }


            for (int t = 0; t < neighbor_count; t++) {
                int idxA = j;
                int idxB = E[j][1 + t]-1;
                int idxC = E[j][1 + ((t + 1) % neighbor_count)]-1;

                double v1x = X[idxB] - X[idxA], v1y = Y[idxB] - Y[idxA], v1z = Z[idxB] - Z[idxA];
                double v2x = X[idxC] - X[idxA], v2y = Y[idxC] - Y[idxA], v2z = Z[idxC] - Z[idxA];
                double nx_t = v1y * v2z - v1z * v2y;
                double ny_t = v1z * v2x - v1x * v2z;
                double nz_t = v1x * v2y - v1y * v2x;
                double nlen = sqrt(nx_t * nx_t + ny_t * ny_t + nz_t * nz_t);
                if (nlen < 1e-14) continue;
                nx_t /= nlen; ny_t /= nlen; nz_t /= nlen;

                double inner_F = 0.0;

                for (int k = 0; k < 7; ++k) {
                    double rxg = X[i] - XG[j][t][k];
                    double ryg = Y[i] - YG[j][t][k];
                    double rzg = Z[i] - ZG[j][t][k];

                    double r = sqrt(rxg * rxg + ryg * ryg + rzg * rzg);

                    double num = nx_t * rxg + ny_t * ryg + nz_t * rzg;
                    if (r < 1e-12) continue;
                    inner_F += - eta1[k] * wk[k] * (num / (r * r * r));
                }

                Fij += AA[j][t] * inner_F;
            }

            F[i][j] = 1.0 / (4.0 * PI) * Fij;
        }
    }  
     
OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += F[i][j];
        }
        F[i][i] = 1 - sum;
    }

    Gdiag = new double[N];

    
OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        double Gii = 0.0;
        int neighbor_count = E[i][0];
        if (neighbor_count < 3 || neighbor_count > MAX_TRI) {
            fprintf(stderr, "Invalid neighbor_count=%d at element %d\n", neighbor_count, i);
            status = 1;
            goto cleanup;
        }

        for (int t = 0; t < neighbor_count; ++t) {
            int idxA = i;
            int idxB = E[i][1 + t] - 1;
            int idxC = E[i][1 + ((t + 1) % neighbor_count)] - 1;
            if (idxB < 0 || idxB >= N || idxC < 0 || idxC >= N) {
                fprintf(stderr, "Invalid neighbor index at element %d\n", i);
                continue;
            }

            double ABx = X[idxB] - X[idxA];
            double ABy = Y[idxB] - Y[idxA];
            double ABz = Z[idxB] - Z[idxA];

            double ACx = X[idxC] - X[idxA];
            double ACy = Y[idxC] - Y[idxA];
            double ACz = Z[idxC] - Z[idxA];

            double LAB = sqrt(ABx * ABx + ABy * ABy + ABz * ABz);
            double LAC = sqrt(ACx * ACx + ACy * ACy + ACz * ACz);
            if (LAB < 1e-12 || LAC < 1e-12) continue;

            double cos_alpha = (ABx * ACx + ABy * ACy + ABz * ACz) / (LAB * LAC);
            if (cos_alpha > 1.0)  cos_alpha = 1.0;
            if (cos_alpha < -1.0) cos_alpha = -1.0;
            double alpha = acos(cos_alpha);

            double BCx = X[idxC] - X[idxB];
            double BCy = Y[idxC] - Y[idxB];
            double BCz = Z[idxC] - Z[idxB];
            double lt = sqrt(BCx * BCx + BCy * BCy + BCz * BCz);
            if (lt < 1e-12) continue;

            double BAx = X[idxA] - X[idxB];
            double BAy = Y[idxA] - Y[idxB];
            double BAz = Z[idxA] - Z[idxB];
            double LBA = sqrt(BAx * BAx + BAy * BAy + BAz * BAz);
            double LBC = lt;
            if (LBA < 1e-12 || LBC < 1e-12) continue;

            double cos_beta = (BAx * BCx + BAy * BCy + BAz * BCz) / (LBA * LBC);
            if (cos_beta > 1.0)  cos_beta = 1.0;
            if (cos_beta < -1.0) cos_beta = -1.0;
            double beta = acos(cos_beta);

            double arg = tan(0.5 * (alpha + beta)) / tan(0.5 * beta);
            if (arg <= 0.0) continue;

            double term = (AA[i][t] / lt) * log(arg);
            Gii += term;
        }

        Gdiag[i] = Gii / (4.0 * PI);
    }

    b = new double[N];

OMP_PARALLEL_FOR
    for (int i = 0; i < N; i++) {
        b[i] = 0.0;
    }
    
OMP_PARALLEL_FOR
    for (int i = 0; i < N; ++i) {
        double sumRow = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                b[i] += Gdiag[j] * q[j];
                continue;
            }
            double Gij = 0.0;
            int neighbor_count_j = E[j][0];
            if (neighbor_count_j < 3 || neighbor_count_j > MAX_TRI) {
                fprintf(stderr, "Invalid neighbor_count=%d at element %d\n", neighbor_count_j, j);
                continue;
            }

            for (int t = 0; t < neighbor_count_j; ++t) {
                int idxA = j;
                int idxB = E[j][1 + t] - 1;
                int idxC = E[j][1 + ((t + 1) % neighbor_count_j)] - 1;

                if (idxB < 0 || idxB >= N || idxC < 0 || idxC >= N) {
                    fprintf(stderr, "Invalid neighbor index at element %d\n", j);
                    continue;
                }

                double inner_G = 0.0;
                for (int k = 0; k < 7; ++k) {
                    double rx = X[i] - XG[j][t][k];
                    double ry = Y[i] - YG[j][t][k];
                    double rz = Z[i] - ZG[j][t][k];
                    double r = sqrt(rx * rx + ry * ry + rz * rz);
                    
                    if (r < 1e-12) continue;
                    inner_G += eta1[k] * wk[k] / r; 
                }

                inner_G *= AA[j][t];
                Gij += inner_G;
            }

            Gij *= 1.0 / (4.0 * PI);

            sumRow += (Gij * q[j]);
        }
        b[i] += sumRow;
    }

    u = bicgstab(F, b, N);
    
    if (!u) {
        fprintf(stderr, "Solver failed: u is null.\n");
        status = 1;
        goto cleanup;  
    }
    // u is now available for validation / EOC

    file3 = fopen("data1.dat", "w");
    if (!file3) { perror("data1.dat"); status = 1; goto cleanup; }

    for (int i = 0; i < N; i++) {
        fprintf(file3, "%lf\t%lf\t%.20lf\n", B[i] * 180 / PI, L[i] * 180 / PI, u[i]);
    }

    //EOC PART

    if (RUN_EOC) {
        const bool WRITE_FILES = true; // set false if you only want to read existing files

        if (WRITE_FILES) {
            if (N == 902) {
                if (!write_error_file("data902.dat", u, 902)) {status = 1; goto cleanup;}
            } else if (N == 3602) {
                if (!write_error_file("data3602.dat", u, 3602)) {status = 1; goto cleanup;}
            } else {
                fprintf(stderr, "EOC write requested but N=%d is not supported.\n", N);
                status = 1; 
                goto cleanup;
            }
        }

        double err902  = compute_rmse_from_file("data902.dat",  902);
        double err3602 = compute_rmse_from_file("data3602.dat", 3602);

        if (err902 > 0.0 && err3602 > 0.0) {
            double eoc = log(err902 / err3602) / log(2.0);
            printf("\n=== EOC VALIDATION ===\n");
            printf("Dataset sizes: 902 vs 3602\n");
            printf("RMSE(902)  = %.6e\n", err902);
            printf("RMSE(3602) = %.6e\n", err3602);
            printf("EOC        = %.6f\n", eoc);
            printf("======================\n\n");
        } else {
            printf("EOC: error files not available yet (run both N=902 and N=3602).\n");
            status = 1; 
            goto cleanup;
        }
    }

// ---- cleanup ----
cleanup:
    if (file)  { fclose(file);  file  = nullptr; }
    if (file2) { fclose(file2); file2 = nullptr; }
    if (file3) { fclose(file3); file3 = nullptr; }


    if (B) {delete[] B; B = nullptr;}
    if (L) {delete[] L; L = nullptr;}
    if (H) {delete[] H; H = nullptr;}
    if (q) {delete[] q; q = nullptr;}
    if (tmp) {delete[] tmp; tmp = nullptr;}

    if(E){
        for (int i = 0; i < N; i++) delete[] E[i];
        delete[] E;
        E = nullptr;
    }
    if(AA){
        for (int i = 0; i < N; i++) delete[] AA[i];
        delete[] AA;
        AA = nullptr;
    }

    if (XG && YG && ZG) {
        for (int i = 0; i < N; i++) {
            if (XG[i] && YG[i] && ZG[i]) {
                for (int j = 0; j < MAX_TRI; j++) {
                    delete[] XG[i][j];
                    delete[] YG[i][j];
                    delete[] ZG[i][j];
                }
                delete[] XG[i];
                delete[] YG[i];
                delete[] ZG[i];
            }
        }
        delete[] XG;
        delete[] YG;
        delete[] ZG;

        XG = nullptr;
        YG = nullptr;
        ZG = nullptr;
    }

    if (F) {
        for (int i = 0; i < N; i++) {
            delete[] F[i];
        }
        delete[] F;
        F = nullptr;
    }

    if (Gdiag) {delete[] Gdiag; Gdiag = nullptr;}

    if (b) {delete[] b; b = nullptr;}

    if (u) {delete[] u; u = nullptr;}

    if (nx) {delete[] nx; nx = nullptr;}
    if (ny) {delete[] ny; ny = nullptr;}
    if (nz) {delete[] nz; nz = nullptr;}
    if (X) {delete[] X; X = nullptr;}
    if (Y) {delete[] Y; Y = nullptr;}
    if (Z) {delete[] Z; Z = nullptr;}

    return status;
}