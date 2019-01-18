from functools import reduce

def flatten_list(unflat):
    # check if list needs to be flattened
    # in case it flattens it to pure string

    if len(unflat) < 2:
        return unflat
    else:
        flatten_list = reduce(lambda x, y: x+y, list(unflat))
        return flatten_list

def clean_single_list(unclean_list):
    clean_list = []
    for element in unclean_list:
        if '.jpg' in element or '.png' in element or '.jpeg' in element:
            clean_list.append(element)
        else:
            # if file does not follow type, ignore
            pass

    return clean_list


def clean_multiple_list(unclean_list):
    clean_list = []
    length = len(unclean_list)

    for i in range(length):
        sub_list = []
        for item in unclean_list[i]:
            if '.jpg' in item or '.png' in item or '.jpeg' in item:
                sub_list.append(item)
            else:
                # if file does not follow type, ignore
                pass
        clean_list.append(sub_list)

    return clean_list


# testing purposes
if __name__ == "__main__":
    pass