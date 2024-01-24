import re
import matplotlib.pyplot as plt


def extract_loss_and_epoch(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Define the pattern to match the desired format
    pattern = r"'eval_loss': ([0-9.]+), .* 'epoch': ([0-9.]+)"

    # Use regex to find all matches
    matches = re.findall(pattern, log_content)

    # Extract eval_loss and epoch values
    eval_loss_values = [float(match[0]) for match in matches]
    epoch_values = [float(match[1]) for match in matches]

    return eval_loss_values, epoch_values

def plot_loss_vs_epochs(eval_loss_values, epoch_values):

    plt.plot(epoch_values, eval_loss_values,)
    plt.title('Validation Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig('eval_loss.png', dpi=300)

# Example usage
log_file_path = 'log_GPT2_50_epochs.out'
eval_loss_values, epoch_values = extract_loss_and_epoch(log_file_path)

# Print the extracted values
for eval_loss, epoch in zip(eval_loss_values, epoch_values):
    print(f'Eval Loss: {eval_loss}, Epoch: {epoch}')

# Plot the validation loss vs epochs
plot_loss_vs_epochs(eval_loss_values, epoch_values)