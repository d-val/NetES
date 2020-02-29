from multiprocessing import Queue, Process
import math

def mp_factorizer(arglist, ids, nprocs, f):
    def worker(arglist, out_q):
        """ The worker function, invoked in a process. 'nums' is a
            list of numbers to factor. The results are placed in
            a dictionary that's pushed to a queue.
        """
        results = []
        for n, args in zip(ids, arglist):
            results.append(f(*args))
        print("[master][multi] putting results on queue...")
        out_q.put(results)

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    out_q = Queue()
    chunksize = int(math.ceil(len(arglist) / float(nprocs)))
    print("chunksize:", chunksize)
    procs = []

    for i in range(nprocs):
        p = Process(
            target=worker,
            args=(arglist[chunksize * i:chunksize * (i + 1)],
                  out_q))
        procs.append(p)
        print("[master][multi] created process...")
        p.start()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    results = []
    for i in range(nprocs):
        print("[master][multi] getting results from queue...")
        results += out_q.get()
        print("[master][multi] results gotten:", len(results))

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    return results
