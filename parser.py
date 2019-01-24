import argparse

'''
Goal of this file is to provide a CLI interface that allows for
specific commands to speed up development
'''

def init_parse():
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args.accumulate(args.integers))

    return 0



if __name__ == "__main__":
    init_parse()
    pass