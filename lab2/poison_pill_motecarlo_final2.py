import multiprocessing
import time
import random
from ipython_genutils.py3compat import xrange
from math import pi

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        """
        Handles task from queue.
        """
        proc_name = self.name # from Process
        while True:
            next_task = self.task_queue.get() # multiprocessing.JoinableQueue() init from Consumer, get task obj
            if next_task is None: # poision pill
                print(f'Exiting {proc_name}')

                self.task_queue.task_done()
                break
            print(proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)

class Task():
    def __init__(self, a , b):
        self.a = a
        self.b = b

    def __call__(self):
        time.sleep(0.1)  # pretend to take some time to do the work
        random.seed(self.b)
        print("Hello from a worker")
        s = 0
        for i in range(self.a): #args ?
            x = random.random()
            y = random.random()
            #print(f'x {x}, y {y}')
            if x ** 2 + y ** 2 <= 1.0:
                s += 1

        return s



if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue() # like a list

    # Start consumers
    num_consumers = 4


    # multiprocessing.cpu_count()

    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]

    list(worker.start() for worker in consumers)
    # start() Start the process’s activity.
    # This must be called at most once per process object.
    # It arranges for the object’s run() method to be invoked in a separate process.


    # Enqueue jobs
    pi_est = 0
    accuracy = 0.001 # format ok 99% = 0.01 99.9 0.001

    # 0.001

    num_of_predictions = 1000
    n = 0
    print(f' abs val {abs(pi - pi_est)}')
    total_result = 0
    count_tasks = 0
    inc_rnd = 0
    while abs(pi - pi_est) > accuracy:
        inc_rnd += 1
        tasks.put(Task(a=num_of_predictions, b=inc_rnd))
        tasks.join()
        count_tasks += num_consumers
        result = results.get()
        total_result += result
        print('Result:', total_result)
        n += num_of_predictions
        pi_est = (4.0 * total_result) / n
        print(f'pi estimate = {pi_est}')
        print(f'steps {n}')
        print(" Steps\tSuccess\tPi est.\tError")
        print("%6d\t%7d\t%1.5f\t%1.5f" % (n, total_result, pi_est, pi - pi_est))

        #n = 1000 # comp per task/

    for i in range(num_consumers):
        tasks.put(None)  # Terminate

    print(f'number of tasks {count_tasks}')
