from pprint import pprint


def generate():

    usernames = [
            'admin',
            'admin2',
            'admin3'
            ]

    passwords = [
            'password',
            'password2',
            'password3'
            ]


    # means that the list is of the same length, and we can match the return  values
    if len(usernames) == len(passwords):
        account_dict = {}
        for i in range(len(usernames)):
            account_dict[usernames[i]] = passwords[i]
        
        return account_dict
    else:
        return None




if __name__ == "__main__":
    pprint(generate())
