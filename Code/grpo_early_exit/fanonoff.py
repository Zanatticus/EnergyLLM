import subprocess
import time
import sys

def initialize_fan():
    try:
        command = "sudo jetson_clocks --fan"

        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\Fan turn on")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)
        
def fan_on():
    try:
        command = "sudo sh -c 'echo 128 > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1'"

        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\Fan turn on")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)


def fan_off():
    try:
        command = "sudo sh -c 'echo 0 > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1'"
        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            capture_output=True,
            text=True,
            check=True
        )       
        print(f"\Fan turn off")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_fan()
    time.sleep(10)
    fan_on()
    time.sleep(10)
    fan_off()
    time.sleep(10)
    fan_on()


