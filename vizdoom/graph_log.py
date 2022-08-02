import matplotlib.pyplot as plt
import json
import argparse
import os


def run(args):
	log_rows = []
	with open(args.in_logfile) as f:
		for line in f:
			row = json.loads(line)
			log_rows.append(row)
	log_rows = [
		row
		for row in log_rows if args.max_batch is None or
		row['batch'] <= args.max_batch]
	episodes = [row['batch'] for row in log_rows]
	if args.y_axis not in log_rows[0]:
		print(f'y_axis {args.y_axis} doesnt exist')
		print('available values', log_rows[0].keys())
		return
	values = [row[args.y_axis] for row in log_rows]
	# plt.plot(episodes, values)
	# plt.savefig('vizdoom/graph.png')
	if args.average_over == 1:
		plt.plot(episodes, values)
	else:
		episodes_avg = []
		values_avg = []
		# average_over = 50
		num_batches = len(episodes) // args.average_over
		for b in range(num_batches):
			b_start = b * args.average_over
			b_end = b_start + args.average_over
			avg_episode = sum(episodes[b_start:b_end]) / args.average_over
			_values = [float(v) for v in values[b_start:b_end]]
			avg_value = sum(_values) / args.average_over
			episodes_avg.append(avg_episode)
			values_avg.append(avg_value)
		plt.plot(episodes_avg, values_avg)
	plt.xlabel('batch')
	plt.ylabel(args.y_axis)
	image_filepath = args.out_imagefile.format(y_axis=args.y_axis)
	plt.savefig(image_filepath)
	dirname = os.path.dirname(image_filepath)
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	if os.uname()[0] == 'Darwin':
		os.system(f'open {image_filepath}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-logfile', type=str, default='log.txt')
	parser.add_argument('--out-imagefile', type=str, default='tmp/graph_{y_axis}.png')
	parser.add_argument('--max-batch', type=int)
	parser.add_argument('--y-axis', default='reward')
	parser.add_argument('--average-over', type=int, default=1)
	args = parser.parse_args()
	run(args)
