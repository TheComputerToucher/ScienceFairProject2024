#!/usr/bin/env python3
""" This module is a config utillity that allows you to modify the CSV path and the Epoch count"""
import json

DEFAULT_DATA = '{\n\t"epochs": 400,\n\t"data_path": "./data.csv"}\n'

config = json.loads(DEFAULT_DATA)

try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    print("config.json not found")
except json.decoder.JSONDecodeError:
    print("Invalid config.json, using default data")

ACTIVE = True

print("This is the configuration tool for my Science Fair Project")
print("""Please enter your command below, the commands are:
      - epoch: allows you to update the epoch count
      - path: allows you to update the path to the data csv file
      - save: saves your changes
      - exit: exits and saves
      - exitnosav: exits without saving
      - help: prints this message""")

while ACTIVE:
    cmd = input("> ")
    match cmd:
        case "exit":
            ACTIVE = False
            print("Goodbye")
        case "exitnosav":
            confirm = input("Are you sure? Enter y for yes and anything else for no. > ")
            if confirm == "y":
                ACTIVE = False
                print("Goodbye")
            else:
                print("Not exiting")
        case "epoch":
            new_count = input("Enter the new epoch count> ")
            try:
                new_count_num = int(new_count)
                config["epochs"] = new_count_num
            except TypeError:
                print("That's not a number, try again")
        case "save":
            data = json.dumps(config)

            with open("config.json", "w", encoding="utf-8") as f:
                f.write(data)
        case default:
            print(f"Invalid command: {cmd}")
