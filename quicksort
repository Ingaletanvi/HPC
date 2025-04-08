#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Swap two elements in an array
void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

// In-place quicksort using indices
void quicksort(int* arr, int left, int right) {
    if (left >= right) return;

    int pivot = arr[left + (right - left) / 2];
    int i = left;
    int j = right;

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            swap(arr, i, j);
            i++;
            j--;
        }
    }

    if (left < j) quicksort(arr, left, j);
    if (i < right) quicksort(arr, i, right);
}

// Merge two sorted arrays
int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0, j = 0, k = 0;

    while (i < n1 && j < n2) {
        if (arr1[i] < arr2[j])
            result[k++] = arr1[i++];
        else
            result[k++] = arr2[j++];
    }
    while (i < n1)
        result[k++] = arr1[i++];
    while (j < n2)
        result[k++] = arr2[j++];

    return result;
}

int main(int argc, char* argv[]) {
    int number_of_elements;
    int* data = NULL;
    int chunk_size, own_chunk_size;
    int* chunk;
    FILE* file = NULL;
    double start_time, end_time;
    MPI_Status status;

    if (argc != 3) {
        printf("Usage: mpirun -np <num_procs> ./a.out input.txt output.txt\n");
        exit(EXIT_FAILURE);
    }

    int number_of_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Step 1: Read input file (only by rank 0)
    if (rank == 0) {
        file = fopen(argv[1], "r");
        if (file == NULL) {
            printf("Error opening input file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fscanf(file, "%d", &number_of_elements);
        data = (int*)malloc(number_of_elements * sizeof(int));

        for (int i = 0; i < number_of_elements; i++) {
            fscanf(file, "%d", &data[i]);
        }

        fclose(file);
    }

    // Broadcast number of elements to all processes
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate chunk size
    chunk_size = (number_of_elements + number_of_processes - 1) / number_of_processes;

    // Allocate memory for each chunk
    chunk = (int*)malloc(chunk_size * sizeof(int));

    // Scatter data
    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine actual size of this chunk (may be less on last process)
    own_chunk_size = (number_of_elements >= chunk_size * (rank + 1))
        ? chunk_size
        : (number_of_elements - chunk_size * rank);

    if (data != NULL) {
        free(data);
        data = NULL;
    }

    MPI_Barrier(MPI_COMM_WORLD); // synchronize before timing
    start_time = MPI_Wtime();

    // Sort each local chunk
    quicksort(chunk, 0, own_chunk_size - 1);

    // Step 2: Tree-based merge
    for (int step = 1; step < number_of_processes; step *= 2) {
        if (rank % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break;
        }

        if (rank + step < number_of_processes) {
            int received_chunk_size = (number_of_elements >= chunk_size * (rank + 2 * step))
                ? chunk_size * step
                : (number_of_elements - chunk_size * (rank + step));

            int* received_chunk = (int*)malloc(received_chunk_size * sizeof(int));
            MPI_Recv(received_chunk, received_chunk_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);

            data = merge(chunk, own_chunk_size, received_chunk, received_chunk_size);

            free(chunk);
            free(received_chunk);

            chunk = data;
            own_chunk_size += received_chunk_size;
        }
    }

    end_time = MPI_Wtime();

    // Step 3: Output result
    if (rank == 0) {
        file = fopen(argv[2], "w");
        if (file == NULL) {
            printf("Error opening output file.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(file, "%d\n", own_chunk_size);
        for (int i = 0; i < own_chunk_size; i++) {
            fprintf(file, "%d ", chunk[i]);
        }
        fclose(file);

        printf("\nSorted array written to output file: %s\n", argv[2]);
        printf("Time taken to sort %d elements using %d processes: %f seconds\n",
               number_of_elements, number_of_processes, end_time - start_time);
    }

    free(chunk);
    MPI_Finalize();
    return 0;
}
