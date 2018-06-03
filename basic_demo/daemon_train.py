import os
import daemon

from train import main

with daemon.DaemonContext():
    print('Daemon started with PID {}'.format(os.getpid()))

    with open('pid') as f:
        f.write(os.getpid())
        print('PID saved to file')

    print('Running main program')
    main()
