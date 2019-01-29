from flask import Flask, render_template, redirect, request, sessions
import auth.accounts as accounts
import os
from parser import parser

from pprint import pprint

# def init_server():
# define the flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'livetest')

# this resources will be stored as the user uses the program throughout
resources = []

# initialize table_details
attendance_data = parser()

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

@app.route("/")
def index():
    # pprint(table_details)
    return render_template('index.html', attendance_data=parser())
    # return "Hello World!"

# use this routes for displaying images
@app.route('/<image>')
def profile(image):
    path_to_image = os.path.join(app.config['UPLOAD_FOLDER'], image)
    return render_template('image.html', user_image=path_to_image)


if __name__ == "__main__":
    # initialize the flask server
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0')
