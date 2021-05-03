from mrjob.job import MRJob
from mrjob.step import MRStep
import statistics
import time

class Problem_1a(MRJob):
    def mapper(self, _, line):
        c=0
        for value in line.split('\t'):
            c+=1
            if c%3 == 0: # yield <value> from each line
                yield "None",float(value)


    def combiner(self, _, value):
        count = 0
        values = 0
        for idx, v in enumerate(value):
            count += 1
            values += v
        yield "None", (values,count)


    def reducer(self, _, tuple):
        counts = 0
        values = 0
        for idx,tup in enumerate(tuple):
            values += tup[0]
            counts += tup[1]
        avg = values/counts
        yield "Avg", avg


if __name__ == '__main__':

    begin_time = time.time()
    Problem_1a.run()
    print("Execution time in seconds ", time.time() - begin_time)