import os
import csv


def write_episode_data_to_csv(name, episode_reward, episode_number, steps, exit_reason):
    fieldnames = ["episode", "total_reward", "steps", "exit_reason"]
    episode_data = [episode_number, episode_reward, steps, exit_reason]

    file_path = f"../models/logs/{name}.csv"

    with open(file_path, "a+") as f:
        writer = csv.writer(f)
        # If the file is empty
        if os.stat(file_path).st_size == 0:
            # Write the header
            writer.writerow(fieldnames)

        writer.writerow(episode_data)
