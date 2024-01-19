#!/usr/bin/env python3
# A barebones configuration utillity that allows you to modify the CSV path and the Epoch count
import json

config = None

default_data = '{\n\t"epochs": 400,\n\t"data_path": "./data.csv"}'

try:
    with open("config.json", "r") as f:
        config = json.load(f)
except Exception as e:
    print(f"Encountered error: {e}")
    config = json.loads(default_data)
        
        
active = True

print("This is the configuration tool for my Science Fair Project")
print("Please enter your command below, the commands are:")
print("""- epoch: allows you to update the epoch count
      -  path: allows you to update the path to the data csv file
      - save: saves your changes
      - exit: exits and saves
      - exitnosav: exits without saving
      - help: prints this message""")

while active:
    cmd = input("> ")
    match cmd:
        case "exit":
            active = False
            print("Goodbye")
        case "exitnosav":
            confirm = input("Are you sure? Enter y for yes and anything else for no. > ")
            if confirm == "y":
                active = False
                print("Goodbye")
            else:
                print("Not exiting")
        case "epoch":
            new_count = input("Enter the new epoch count> ")
            new_count_num = 0
            try:
                new_count_num = int(new_count)
                config["epochs"] = new_count_num
            except TypeError:
                print("That's not a number, try again")
        case "save":
            data = json.dumps(config)
            
            with open("config.json", "w") as f:
                f.write(data)
        case default:
            print(f"Invalid command: {cmd}")


