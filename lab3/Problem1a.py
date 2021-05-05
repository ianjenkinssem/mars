from mrjob.job import MRJob
from mrjob.step import MRStep
import math

class CalculateStats(MRJob):

    def mapper(self, _, line):
        words = line.split('\t')
        grp = int(words[1])
        yield "None", float(words[2])  # Yield <value>

    def combiner(self, _, value):
        count = 0
        values = 0
        sumofsquares = 0
        listofvalues = []

        for idx, v in enumerate(value):
            count += 1
            values += v
            sumofsquares += v**2
            listofvalues.append(v)  # Group all the supplied values in a list

        # Yield (sum of values, count, sumofsquares, min, max) tuple for each mapper
        yield "None", (values,count,sumofsquares,min(listofvalues),max(listofvalues))

    def reducer(self, _, tuple):
        counts = 0
        values = 0
        sumofsquares = 0
        listofminvalues = []
        listofmaxvalues = []

        for idx,tup in enumerate(tuple): # Iterating through Iterator object
            values += tup[0]  # Calculate sum of values
            counts += tup[1]  # Calculate total count
            sumofsquares += tup[2] # Aggregate total sum of squares of all values
            listofminvalues.append(tup[3])  # Group all minimum values from each combiner
            listofmaxvalues.append(tup[4])  # Group all maximum values from each combiner

        # Calculate Mean
        mean = values/counts

        # Calculate std deviation
        stddev = math.sqrt(sumofsquares/counts - mean**2)

        # Yield Mean, Std deviation,Min, Max values
        yield "Mean", mean
        yield "Stddev",stddev
        yield "Min value", min(listofminvalues)
        yield "Max value", max(listofmaxvalues)

    def steps(self):
        return [MRStep(mapper=self.mapper,
                       combiner=self.combiner,
                       reducer=self.reducer)]

#if __name__ == '__main__':
    #CalculateStats.run()
