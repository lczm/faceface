from flask import Flask, render_template, redirect, request, sessions
import auth.accounts as accounts


# def init_server():
# define the flask app
app = Flask(__name__)

# this resources will be stored as the user uses the program throughout
resources = []

'''
initialize the flask routes
'''

# this route is for the singin page
# start off with a signin page
@app.route("/signin", methods=["GET", "POST"])
def signin():
    # the method used is post
    if request.method == 'POST':

        # generate on the fly the usernames and passwords
        account_dict = accounts.generate()
        
        # get the username and password
        # if the checkbox is ticked, add a cookie to the browser

        username = request.form['username']
        password = request.form['userpassword']

        print('Username : ', username)
        print('Password : ', password)

        if username in account_dict.keys():
            if password == account_dict[username]:
                # if they get to this point i tmeans that the login is successful
                # redirect them to the homepage
                return redirect('./')
            else:
                return render_template('login.html')
        else:
            return render_template('login.html')

        return render_template('login.html')
    elif request.method == 'GET':
        return render_template('login.html')
    else:
        print('getting here is bad')
        return render_template('error.html')

@app.route("/test", methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        print('this is a post')
        print(request.form['test'])
        return "test world"
    else:
        return "Test world"

@app.route("/")
def index():
    return "Hello World!"



if __name__ == "__main__":
    # initialize the flask server
    app.run(debug=True, host='0.0.0.0')
