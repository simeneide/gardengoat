# gardengoat


## Installation of raspberry pi

### Install os + wifi
1. Download full raspberian at  https://www.raspberrypi.org/downloads/raspbian/
2. Flash sd card
3. Create file "ssh" in boot directory
4. Connect pi to network
5. Set configs in `sudo raspi-config`
    a. Set wifi ssid+pwd
    b. Set host name to gardengoat
6. Clone this repository: `git clone https://github.com/simeneide/gardengoat.git`
	
### Install gps module
1. Set the following in raspi-config:https://learn.adafruit.com/adafruit-ultimate-gps-hat-for-raspberry-pi/pi-setup

### Installing motor driver
Following [this](https://learn.adafruit.com/adafruit-dc-and-stepper-motor-hat-for-raspberry-pi/installing-software) instructions to install driver to motor:
1. Enable I2C: https://learn.adafruit.com/adafruits-raspberry-pi-lesson-4-gpio-setup/configuring-i2c
2. Install library: `sudo pip3 install adafruit-circuitpython-motorkit`


### Other packages
- apriltags (https://pypi.org/project/apriltags/) (need `apt-get update && apt-get install cmake`)
- Installed by building here: `https://github.com/AprilRobotics/apriltag`

pip3 install jupyterlab
apt-get install emacs