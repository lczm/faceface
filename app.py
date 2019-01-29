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
@app.route("/index")
def index():
    # initialize table_details
    attendance_data = parser()

    # pprint(table_details)
    return render_template('index.html', attendance_data=parser())

# use this routes for displaying images
@app.route('/<image>')
def image(image):
    path_to_image = os.path.join(app.config['UPLOAD_FOLDER'], image)
    return render_template('image.html', user_image=path_to_image)

# this route is for displaying the user
@app.route('/user/<username>')
def user(username):
    userdata = []

    # initialize table_details
    attendance_data = parser()
    # get the [1] index from every part of attendance_data
    user_names = []
    user_index = []
    for data in attendance_data:
        user_names.append(data[1])

    if username in user_names:
        for i in range(len(user_names)):
            if user_names[i] == username:
                user_index.append(i)

    specific_data = []
    # get the data for the indexes given

    for i in range(len(user_index)):
        specific_data.append(attendance_data[user_index[i]])

    pprint(specific_data)

    user_attendance = len(specific_data)
    total_attendance = len(attendance_data)

    user_data = []

    format_total_attendance = "{} / {}".format(user_attendance, total_attendance)
    userdata.append(format_total_attendance)

    # per class attendance is the same as format total attendance as there is only one class
    userdata.append(format_total_attendance)

    # get the specific user data here
    return render_template('user.html', user_data=userdata)


if __name__ == "__main__":
    # initialize the flask server
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0')

