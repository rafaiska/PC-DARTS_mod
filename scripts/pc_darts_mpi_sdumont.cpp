# include <cstdlib>
# include <ctime>
# include <sstream>
# include <mpi.h>

#define WORKER_CMD "singularity exec -B ./mount:/workspace/mount --nv pc_darts_original.sif /workspace/mount/run_train_mx.sh"

int main ( int argc, char *argv[] );
void timestamp ( );

void runWorker(int id)
{
    std::stringstream ss;
    std::string command;
    ss << WORKER_CMD << ' ';
    ss << id;
    system(ss.str().c_str());
}


int main ( int argc, char *argv[] )
{
    int id;
    int ierr;
    int p;
    MPI_Group world_group_id;

    ierr = MPI_Init ( &argc, &argv );

    if ( ierr != 0 )
    {
        std::cout << "\n";
        std::cout << "COMMUNICATOR_MPI - Fatal error!";
        std::cout << "  MPI_Init returned nonzero ierr.\n";
        exit ( 1 );
    }

    ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );
    ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );

    runWorker(id);

    ierr = MPI_Finalize ( );

    return 0;
}
