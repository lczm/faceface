import csv
from pprint import pprint

'''
goal of this file is to parse through the logging csv, 'log.csv'
and return an appropriate datatype for the other modules
'''

def parser(reverse=False):
    with open ('./log.csv', 'r') as file:
        return_list = []

        # columns in order of their indexes
        # 0 -> path to the file
        # 1 -> fail/pass, name
        # 2 -> date and time at which the time is taken
        # 3 -> location

        '''
        for 0 -> path to the file
        only require the filename, thanks to the <image> route for
        the api built in
        '''

        data = csv.reader(file)

        for row in data:
            # can filter out the new 0
            # filter_out = row[0].split('/')
            # print(filter_out)
            # print(filter_out[-1])
            row[0] = row[0].split('/')[-1]
            return_list.append(row)

        if reverse==True:
            for number, row in reverse(enumerate(data, start=1)):
                row[0] = row[0].split('/')[-1]
                return_list.append(row)

    # return return_dict
    return return_list


if __name__ == "__main__":
    # returns a dictionary
    pprint(parser())
