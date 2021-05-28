"""
A very simple script to run BRepNet multiple times and search parameters
"""
import argparse
import copy
from pytorch_lightning import Trainer
import xlsxwriter

from pipeline.running_stats import RunningStats
from models.brepnet import BRepNet
from train.train import do_training

def write_col_headings(worksheet, option1, option2, output):
    """
    Write the headings for the worksheet columns
    """
    worksheet.write(0, 0, "Timestamps")
    worksheet.write(0, 1, option1)
    worksheet.write(0, 2, option2)
    key_list = []
    for i, key in enumerate(output):
        key_list.append(key)
        split_key = key.split("/")[1]
        worksheet.write(0, (2*i)+3, split_key+"_mean")
        worksheet.write(0, (2*i)+4, split_key+"_std")

    return key_list


def write_results_to_workbook(worksheet, option1, option2, results):
    """
    Write out the data to a worksheet
    """
    assert len(results) > 0
    key_list = write_col_headings(worksheet, option1, option2, results[0]["output"])
    for i, result in enumerate(results):
        worksheet.write(i+1, 0, result["timestamps"])
        worksheet.write(i+1, 1, result["option1_value"])
        worksheet.write(i+1, 2, result["option2_value"])
        for j, key in enumerate(key_list):
            worksheet.write(i+1, (2*j)+3, result["output"][key]["mean"])
            worksheet.write(i+1, (2*j)+4, result["output"][key]["std"])


def write_results_to_excel(excel_workbook, option1, option2, results):
    """
    Write the results to an excel workbook
    """
    workbook = xlsxwriter.Workbook(excel_workbook)
    worksheet = workbook.add_worksheet()
    write_results_to_workbook(worksheet, option1, option2, results)
    workbook.close()

def mean_and_std_from_outputs(outputs):
    accumulate = {}
    for output in outputs:
        for key in output:
            if not key in accumulate:
                accumulate[key] = RunningStats()
            accumulate[key].push(output[key])
    mean_and_std = {}
    for key in accumulate:
        mean_and_std[key] = {
            "mean": accumulate[key].mean(),
            "std":accumulate[key].standard_deviation()
        }
    return mean_and_std

def average_multinode_test_results(results):
    accumulate = {}
    for result in results:
        for key in result:
            if not key in accumulate:
                accumulate[key] = 0.0
            accumulate[key] += result[key]
    num_outputs = len(results)
    for key in accumulate:
        accumulate[key] /= float(num_outputs)
    return accumulate

def do_grid_search(base_opts):
    option1 = base_opts.option1
    option1_values = base_opts.option1_values
    option2 = base_opts.option2
    option2_values = base_opts.option2_values

    option1_type = type(getattr(base_opts, option1))
    option2_type = type(getattr(base_opts, option2))

    results = []

    for option1_value in option1_values:
        for option2_value in option2_values:
            opts_to_use = copy.deepcopy(base_opts)
            setattr(opts_to_use, option1, option1_type(option1_value))
            setattr(opts_to_use, option2, option2_type(option2_value))
            outputs_for_mean_and_std = []
            timestamps = ""
            for i in range(base_opts.num_runs_to_average):
                outputs = do_training(opts_to_use)
                multi_node_average = average_multinode_test_results(outputs["test_results"])
                outputs_for_mean_and_std.append(multi_node_average)
                timestamps += outputs["month_day"] + "/" + outputs["hour_min_second"] +";"
            result = {
                "timestamps": timestamps,
                "option1_value": option1_type(option1_value),
                "option2_value": option2_type(option2_value),
                "output": mean_and_std_from_outputs(outputs_for_mean_and_std)
            }
            results.append(result)

            # Write out the workbook after each training so
            # we don't lose anything
            write_results_to_excel(base_opts.excel_workbook, option1, option2, results)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BRepNet.add_model_specific_args(parser)
    parser.add_argument(
        "--num_runs_to_average", 
        type=int, 
        required=True,  
        help="The number of times to train for computing the mean and std"
    )

    parser.add_argument("--option1", type=str, required=True,  help="Name option1 which will be searched")
    parser.add_argument("--option1_values", type=str, nargs='+', required=True,  help="Values for first option")
    parser.add_argument("--option2", type=str, required=True,  help="Name option2 which will be searched")
    parser.add_argument("--option2_values", type=str, nargs='+', required=True,  help="Values for second option")
    parser.add_argument("--excel_workbook", type=str, required=True,  help="Name for the output .xlsx file")
    base_opts = parser.parse_args()
    do_grid_search(base_opts)

    print("Completed train/grid_search.py")