import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.tools import VQAEval

x_bins = [1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 80000, 96000, 128000]
# x_bins = [1000, 2000, 3000, 5000, 9000, 15000, 26000, 44000, 75000]
y_interval = 0.2
vqa = VQAEval()

def is_correct(answer, response):
    response_orig = response
    if isinstance(answer, int):
        if response.isdigit():
            return int(int(response) == answer)

        response = response.lower()
        if response.find('.') != -1:
            response = response.split('.')[0]
            response = response.replace(',', '')
            response = response.strip()

        if response == 'none':
            return 0

        if len(response) != 1:
            print(f"Fail to parse {response_orig}")
            return 0

        return (ord(response) - ord('a')) == answer

    if isinstance(answer, list):
        try:
            response = json.loads(response)
        except Exception as e:
            print(f"Fail to parse {response_orig} Exception: {e}")
            return 0

        match = 0
        for res, ans in zip(response, answer):
            match += res == ans
        return match / len(ans)

    return vqa.evaluate(response, answer)

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f'{args.save_dir}_details', exist_ok=True)

    result_path_list = os.listdir(args.outputs_dir)
    for file_name in result_path_list:
        total = np.zeros((len(x_bins) + 1, int(1 / y_interval)))
        correct = np.zeros((len(x_bins) + 1, int(1 / y_interval)))

        jsonl_file_path = os.path.join(args.outputs_dir, file_name)
        file_path = os.path.join(args.save_dir, file_name.replace('.jsonl', '.png'))
        file_path_pdf = os.path.join(f'{args.save_dir}_details', file_name.replace('.jsonl', '.pdf'))
        file_path_txt = os.path.join(f'{args.save_dir}_details', file_name.replace('.jsonl', '.txt'))

        if os.path.isdir(jsonl_file_path):
            continue

        with open(jsonl_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                x = entry['total_tokens']
                y = entry['position']
                if isinstance(y, list):
                    y = sum(entry['position']) / len(entry['position'])
                else:
                    y = entry['position']

                if y == 1.0:
                    y = 0.99

                z = entry['response']
                answer = entry['answer']

                x_index = np.digitize(x, x_bins)
                y_index = int(y / y_interval)
                total[x_index][y_index] += 1
                correct[x_index][y_index] += is_correct(answer, z)

            result = np.divide(correct, total, out=np.zeros_like(correct), where=total != 0)

        # print(result)
        print(file_name)
        print(total)
        print()

        # # Plot a heatmap for a numpy array:
        uniform_data = result[1:].T
        # print(uniform_data)

        # Define the custom color map
        from matplotlib.colors import LinearSegmentedColormap

        colors = colors = ["#DC143C", "#FFD700", "#3CB371"]  # Red to Yellow to Green
        n_bins = 100  # Discretizes the interpolation into bins
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        ax = sns.heatmap(uniform_data, vmin=0, vmax=1, cmap=cm)

        plt.xticks(ticks=np.arange(uniform_data.shape[1])+0.5, labels=[f'{i / 1000}k' for i in x_bins])
        plt.xticks(rotation=90)

        plt.yticks(ticks=np.arange(uniform_data.shape[0]), labels=[f'{j / (1/y_interval)}' for j in range(int(1/y_interval))])
        plt.yticks(rotation=0)

        plt.savefig(file_path)
        plt.savefig(file_path_pdf)
        plt.clf()

        with open(file_path_txt, 'w') as file:
            file.write(json.dumps(total[1:].T.tolist()) + '\n\n')
            file.write(json.dumps(uniform_data.tolist()) + '\n\n')
            file.write(json.dumps(uniform_data.mean(axis=0).tolist()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualization script for outputs")
    parser.add_argument('--outputs-dir', type=str, default='')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.outputs_dir, 'visualization')
    main(args)
