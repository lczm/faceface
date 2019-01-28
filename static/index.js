// JavaScript source code

function startTime() {
    let today = new Date();  //letiable today gets new Date()

    //Creates an array and allocates each number of the day to its respective name
    let weekday = new Array(7);
    weekday[0] = "Sunday";
    weekday[1] = "Monday";
    weekday[2] = "Tuesday";
    weekday[3] = "Wednesday";
    weekday[4] = "Thursday";
    weekday[5] = "Friday";
    weekday[6] = "Saturday";

    //Gets the day of the week
    let n = weekday[today.getDay()];

    //Gets the date in YYYY MM DD
    let d = today.getDate();
    let month = today.getMonth();
    let y = today.getFullYear();

    //Gets the time in hours, minutes and seconds in a 24 hour clock format
    let h = today.getHours();
    let m = today.getMinutes();
    let s = today.getSeconds();
    m = checkTime(m);
    s = checkTime(s);
    let t = setTimeout(startTime, 0);

    //Displays the day of the week, date and time
    document.getElementById('datetime').innerHTML = n + " | " + d + "/ " + month + 1 + "/ " + y + " | " + h + ":" + m + ":" + s;
}
function checkTime(i) {
    if (i < 10) { i = "0" + i };  // add zero in front of numbers < 10
    return i;
}
