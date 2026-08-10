// MPI availability probe for hellofem.
// Verifies: headers link, MPI_Init/Finalize, communicator, a real collective.
#include <mpi.h>

#include <cstdio>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0, len = 0;
    char name[MPI_MAX_PROCESSOR_NAME] = {0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(name, &len);
    std::printf("rank %d/%d on %s (MPI v%d.%d)\n", rank, size, name, MPI_VERSION, MPI_SUBVERSION);

    const int local = rank + 1;
    int sum = 0;
    MPI_Allreduce(&local, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        std::printf("allreduce sum over %d ranks = %d (expected %d)\n", size, sum, size * (size + 1) / 2);

    MPI_Finalize();
    return sum == size * (size + 1) / 2 ? 0 : 1;
}
