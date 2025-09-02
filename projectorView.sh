#!/bin/bash

# Small delay to ensure network and GUI are fully ready
sleep 10

# Hide mouse cursor immediately
# unclutter -idle 0 &

function check_local_connection() {
    #this is the ip addres of the raspberry pi running the quiz on the robot
    ping -c 1 192.168.137.135 &> /dev/null
    return $?
}


while ! check_local_connection
do
    echo "Waiting for internet connection..."
    sleep 1
done
echo "Internet connection established. Continuing with the rest of the script..."



/usr/bin/chromium-browser --foreground --kiosk --start-maximized --disable-infobars --noerrdialogs http://robotoo-interface.local/projector &


# changes group2
URL="robotoo-interface.local"
CHECK_INTERVAL=30 # check every 30 seconds

while true; do
	if ! ping -c 10  "$URL" &> /dev/null; then # ping sends network packet to specified host, -c 1 sends one ping request, -W 2 waits 2s before time-out
		echo "Website is unreachable! Restarting chromium..."
		killall chromium  #kill all chromium instance
		sleep 2  #wacht 2 seconden
        /usr/bin/chromium-browser --foreground --kiosk --start-maximized --disable-infobars --noerrdialogs http://robotoo-interface.local/projector &
		#/usr/bin/chromium-browser --noerrdialogs --disable-infobars http://robotoo-interface.local/projector &
	fi #end if statement
	# /usr/bin/chromium-browser --noerrdialogs --disable-infobars http://127.0.0.1/demo &
	sleep "$CHECK_INTERVAL"
done &

