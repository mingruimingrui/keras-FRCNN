import os
import sys
import daemon

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import keras_pipeline
    __package__ = "keras_pipeline"

from train import main


if __name__ == '__main__':
    cwd = os.path.dirname(os.path.abspath(__file__))
    log_file = open('daemon.log', 'w+')

    with daemon.DaemonContext(working_directory=cwd, stdout=log_file):
        print('Daemon started with PID {}'.format(os.getpid()))

        with open('pid') as f:
            f.write(os.getpid())
            print('PID saved to file')

        print('Running main program')
        main()
