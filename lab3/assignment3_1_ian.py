from mrjob.job import MRJob, MRStep
import statistics as stat


class MRWordFrequencyCount(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_data,
                   combiner=self.combiner,
                   reducer=self.reducer)]

    def mapper_get_data(self, _, line): # 2 c -> 2 mappers
        w = line.split('\t')
        yield "None", float(w[2])

    def combiner(self, _, values):
        c = 0
        for count, _ in enumerate(values):
            c = count

        yield "None", (sum(values), c+1)

    def reducer(self, _, values):
        c = 0
        total = 0
        for count, val in enumerate(values):
            c = count
            total=val[0]

        yield "AVG", (total, c+1)
        #yield key, stat.mean(values)


if __name__ == '__main__':
    MRWordFrequencyCount.run()