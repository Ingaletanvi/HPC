#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// In-place quicksort
void quicksort(int* arr, int left, int right) {
    if (left >= right) return;
    int pivot = arr[(left + right) / 2], i = left, j = right;
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            int tmp = arr[i];
            arr[i++] = arr[j];
            arr[j--] = tmp;
        }
    }
    quicksort(arr, left, j);
    quicksort(arr, i, right);
}

// Merge two sorted arrays
int* merge(int* a, int na, int* b, int nb) {
    int* res = (int*)malloc((na + nb) * sizeof(int));
    int i = 0, j = 0, k = 0;
    while (i < na && j < nb)
        res[k++] = (a[i] < b[j]) ? a[i++] : b[j++];
    while (i < na) res[k++] = a[i++];
    while (j < nb) res[k++] = b[j++];
    return res;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, *data = NULL;
    if (rank == 0) {
        FILE* f = fopen(argv[1], "r");
        fscanf(f, "%d", &n);
        data = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) fscanf(f, "%d", &data[i]);
        fclose(f);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int chunk = (n + size - 1) / size;
    int* local = (int*)malloc(chunk * sizeof(int));
    MPI_Scatter(data, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = (rank == size - 1) ? n - chunk * rank : chunk;
    quicksort(local, 0, local_size - 1);

    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) != 0) {
            MPI_Send(local, local_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            free(local);
            break;
        }
        if (rank + step < size) {
            int recv_size = (rank + 2 * step <= size) ? chunk * step : n - chunk * (rank + step);
            int* recv = (int*)malloc(recv_size * sizeof(int));
            MPI_Recv(recv, recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int* merged = merge(local, local_size, recv, recv_size);
            free(local); free(recv);
            local = merged;
            local_size += recv_size;
        }
    }

    if (rank == 0) {
        FILE* out = fopen(argv[2], "w");
        fprintf(out, "%d\n", local_size);
        for (int i = 0; i < local_size; i++) fprintf(out, "%d ", local[i]);
        fclose(out);
        printf("Sorted output written to %s\n", argv[2]);
    }

    free(local);
    if (data) free(data);
    MPI_Finalize();
    return 0;
}
