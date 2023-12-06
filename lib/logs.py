import os
import csv


def write_episode_data_to_csv(path, episode_reward, episode_number, steps, exit_reason):
    fieldnames = ["episode", "total_reward", "steps", "exit_reason"]
    episode_data = [episode_number, episode_reward, steps, exit_reason]

    # make sure the path to the file exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        # If the file is empty
        if os.stat(path).st_size == 0:
            # Write the header
            writer.writerow(fieldnames)

        writer.writerow(episode_data)
