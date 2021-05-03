from mrjob.job import MRJob, MRStep
import statistics as stat


class MRWordFrequencyCount(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_data,
                   combiner=self.combiner,
                   reducer=self.reducer)]


    def mapper_get_data(self, _, line): # 2 c -> 2 mappers
        yield "None", float(line.split()[2])

    def combiner(self, key, vaules):
        for _, j  in enumerate(vaules):
            yield "Total", (sum(j), 1)

    def reducer(self, key, values):
        yield key, sum(values)
        #yield key, stat.mean(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()