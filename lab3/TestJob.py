from Problem1a import CalculateStats

import time

if __name__ == '__main__':
    begin_time = time.time()
    mr_job = CalculateStats(args=['--cat-output', 'mist-1000.dat'])

    with mr_job.make_runner() as runner:
        runner.run()
        for key, value in mr_job.parse_output(runner.cat_output()):
            print(key, value)
    print("Execution time in seconds ", time.time() - begin_time)