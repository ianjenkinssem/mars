from mrjob.job import MRJob
import statistics
import time

class Problem_1a(MRJob):

    def mapper(self, _, line):
        words = line.split('\t')
        yield "None", float(words[2])  # Yield <value>

    def combiner(self, _, value):
        count = 0
        values = 0
        for idx, v in enumerate(value):
            count += 1
            values += v
        yield "None", (values,count) # Yield (sum of values, count) tuple for each mapper


    def reducer(self, _, tuple):
        counts = 0
        values = 0
        for idx,tup in enumerate(tuple): # Iterating thru Iterator object
            values += tup[0]  # Retrieve sum of values
            counts += tup[1]  # Retrieve total count
        avg = values/counts
        yield "Avg", avg


if __name__ == '__main__':

    begin_time = time.time()
    Problem_1a.run()
    print("Execution time in seconds ", time.time() - begin_time)