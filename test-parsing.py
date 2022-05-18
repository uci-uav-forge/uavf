# testing file download and parsing without including ROS
import requests

URL = ""

def wp_index():
    file = requests.get(URL)
    mission_list = file.readlines()
    print(mission_list)

def main():
    wp_index()

if __name__ == '__main__':
    main()