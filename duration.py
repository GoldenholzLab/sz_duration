import pandas as pd
from io import StringIO
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import AutoLocator, MultipleLocator

def read_seizuretracker_csv(file_path):
    print(f"Attempting to read file: {file_path}")
    
    # Read the entire file content
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        print("File read successfully")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    # Split the content into sections
    sections = content.split('\n\n')
    print(f"Number of sections found: {len(sections)}")

    # Process Profiles section
    try:
        profiles_section = next(section for section in sections if section.startswith('Profiles'))
        profiles_lines = profiles_section.split('\n')[1:-1]  # Exclude 'Profiles' and 'End Profiles' lines
        profiles_df = pd.read_csv(StringIO('\n'.join(profiles_lines)), skipinitialspace=True)
        print(f"Profiles DataFrame shape: {profiles_df.shape}")
    except Exception as e:
        print(f"Error processing Profiles section: {e}")
        return None, None

    # Process Seizures section
    try:
        seizures_section = next(section for section in sections if section.startswith('Seizures'))
        seizures_lines = seizures_section.split('\n')[1:]  # Exclude 'Seizures' line
        seizures_df = pd.read_csv(StringIO('\n'.join(seizures_lines)), skipinitialspace=True, low_memory=False)
        print(f"Seizures DataFrame shape: {seizures_df.shape}")
    except Exception as e:
        print(f"Error processing Seizures section: {e}")
        return None, None

    # Remove duplicate columns from seizures_df
    seizures_df = seizures_df.loc[:, ~seizures_df.columns.duplicated()]
    print(f"Seizures DataFrame shape after removing duplicates: {seizures_df.shape}")

    # Convert Date_Time to datetime, trying multiple formats
    def parse_date(date_string):
        for fmt in ('%Y-%m-%d %H:%M:%S', '%m/%d/%y %H:%M', '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M'):
            try:
                parsed_date = pd.to_datetime(date_string, format=fmt)
                if pd.Timestamp.min < parsed_date < pd.Timestamp.max:
                    return parsed_date
            except ValueError:
                pass
        print(f"Unable to parse date or date out of valid range: {date_string}")
        return pd.NaT

    print("Converting dates...")
    seizures_df['Date_Time'] = seizures_df['Date_Time'].apply(parse_date)
    print("Date conversion complete")

    # Filter based on date range
    print("Filtering dates...")
    date_mask = (seizures_df['Date_Time'] >= '2007-11-01') & (seizures_df['Date_Time'] <= '2024-07-17')
    seizures_df = seizures_df.loc[date_mask]

    # compute duration in seconds
    seizures_df['duration'] = seizures_df['length_hr']*60*60 + seizures_df['length_min']*60 + seizures_df['length_sec']

    print(f"Seizures DataFrame shape after date filtering: {seizures_df.shape}")

    # Print information about invalid dates
    invalid_dates = seizures_df[seizures_df['Date_Time'].isna()]
    print(f"Number of invalid dates: {len(invalid_dates)}")
    if len(invalid_dates) > 0:
        print("Sample of rows with invalid dates:")
        print(invalid_dates.head())

    # Define seizure type substitutions
    type_substitutions = {
        "Secondarily Generalized": "Focal to bilateral tonic clonic*",
        "Complex Partial": "Focal impaired awareness*",
        "Aura Only": "Focal aware*",
        "Other": "Unknown*",
        "Unknown": "Unknown*",
        "Simple Partial": "Focal aware*"
    }
    # Apply substitutions
    seizures_df['type'] = seizures_df['type'].replace(type_substitutions)

    return profiles_df, seizures_df

def age_cut(seizures_df, profiles_df):
    # Create copies of the DataFrames to avoid SettingWithCopyWarning
    seizures = seizures_df.copy()
    profiles = profiles_df.copy()
    
    # Ensure 'Date_Time' in seizures_df and 'Birth_Date' in profiles_df are datetime
    seizures['Date_Time'] = pd.to_datetime(seizures['Date_Time'], errors='coerce')
    profiles['Birth_Date'] = pd.to_datetime(profiles['Birth_Date'], errors='coerce')
    
    # Merge seizures with profiles on Unlinked_ID
    merged = pd.merge(seizures, profiles[['Unlinked_ID', 'Birth_Date']], on='Unlinked_ID', how='left')
    
    # Calculate age at time of seizure
    merged['Age_at_Seizure'] = (merged['Date_Time'] - merged['Birth_Date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Filter out invalid ages (negative or extremely high)
    merged = merged[(merged['Age_at_Seizure'] >= 0) & (merged['Age_at_Seizure'] <= 120)]
    
    # Separate into child and adult seizures
    child_seizures = merged[merged['Age_at_Seizure'] < 18].drop(columns=['Birth_Date', 'Age_at_Seizure'])
    adult_seizures = merged[merged['Age_at_Seizure'] >= 18].drop(columns=['Birth_Date', 'Age_at_Seizure'])
    
    # Print summary statistics
    print(f"Total seizures: {len(merged)}")
    print(f"Child seizures (0-17 years): {len(child_seizures)}")
    print(f"Adult seizures (18+ years): {len(adult_seizures)}")
    print(f"Seizures excluded due to invalid age: {len(seizures) - len(merged)}")
    id_child = child_seizures['Unlinked_ID'].nunique()
    id_adult = adult_seizures['Unlinked_ID'].nunique()
    print(f"Number of unique patients: {profiles['Unlinked_ID'].nunique()} with {id_child} children and {id_adult} adults")
    overalap = set(child_seizures['Unlinked_ID'].unique()).intersection(set(adult_seizures['Unlinked_ID'].unique()))
    print(f"Overlap between children and adults: {len(overalap)}")
    
    return child_seizures, adult_seizures

def drawGraph(seizures_df, limx=10, agetype='',as_subplots=False,thetype='all',output_file='output_file.csv'):
    if thetype == 'all':
        sub_df = seizures_df.copy()
    else:
        sub_df = seizures_df[seizures_df['type'] == thetype].copy()
    
    # Filter durations greater than 0 and convert to minutes
    filtered_sub = sub_df[sub_df['duration'] > 0].copy()
    durations = filtered_sub['duration'] / 60

    # Sort the durations and calculate the cumulative probabilities
    #print(f"Number of seizures: {len(durations)} for number of patients: {len(sub_df['Unlinked_ID'].unique())}")
    sorted_durations = np.sort(durations)
    cumulative_probs = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    # Find the index of the duration closest to 5 minutes

    #index5 = np.abs(sorted_durations - 5).argmin()
    
    # Get the cumulative probability at that index
    #cumulative_prob5 = cumulative_probs[index5]
    
    index90 = np.abs(cumulative_probs - 0.9).argmin()
    cumulative_prob90 = cumulative_probs[index90]
    prob90_time = sorted_durations[index90]

    # Create the plot
    if as_subplots==False:
        plt.figure(figsize=(4, 6))
    

    if agetype == 'child':
        thecolor = 'blue'
    else:
        thecolor = 'red'

    plt.plot(sorted_durations, cumulative_probs * 100, label=agetype, color=thecolor)
    
    plt.axhline(y=90, linestyle='--', linewidth=2, color='gray')
    plt.axvline(x=prob90_time, linestyle='--', linewidth=2, color=thecolor)
    texty = 20 if thecolor == 'red' else 10
    haChoice = 'left' if thecolor == 'red' else 'right'
    textx = prob90_time + (-1 if thecolor == 'red' else 1)
    if prob90_time > 10:
        textx += (-1 if thecolor == 'red' else 1)
    plt.text(textx, texty, f'{prob90_time:.0f}', ha=haChoice, va='center', color=thecolor)
    #plt.text(prob90_time, cumulative_prob90 * 100, f'{prob90_time}', ha='right', va='bottom')

    #print(f'{cumulative_prob90} % at {prob90_time} minutes for {agetype}')

    txt = f"all,{thetype},{agetype},{len(durations)},{len(filtered_sub['Unlinked_ID'].unique())},{prob90_time:.1f},{np.median(sorted_durations):.1f}"
    print(txt)
    # Open the file in append mode, which creates the file if it does not exist
    with open(output_file, 'a') as file:
        file.write(txt + '\n')

    #print(f"Sz n={len(durations)} pts n=:{len(sub_df['Unlinked_ID'].unique())} cdf90={prob90_time} for {agetype}")
    
    #plt.axvline(x=5, linestyle='--', linewidth=2)
    #plt.text(5, cumulative_prob5 * 100, f'{cumulative_prob5:.0%}', ha='right', va='bottom')
    #print(f'{cumulative_prob5} % at 5minutes for {agetype}')

    # Set tick marks 
    limx = np.max([limx, int(prob90_time*1.1)])


    # Set the x-axis limit to limx minutes
    plt.xlim(0, limx)
    
    # Set the y-axis limit from 0 to 100%
    plt.ylim(0, 100)
    
    # Labels and title
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Cumulative Probability (%)')
    if as_subplots==False:
        plt.title(f'CDF of Seizure Durations {agetype}')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)


    if limx < 10:
        xstep = 1
    elif limx > 30:
        xstep = 10
    elif limx > 20:
        xstep = 5
    else:
        xstep = 2
    plt.xticks(np.arange(0, limx+1, xstep))  # X-axis ticks from 0 to 10, every 1 minute
    plt.yticks(np.arange(0, 101, 10))  # Y-axis ticks from 0% to 100%, every 10%
    
    # Set minor tick marks
    ax = plt.gca()  # Get the current Axes instance
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for x-axis, between the major ticks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for y-axis, between the major ticks
    # Add minor grid lines
    ax.grid(True, linestyle=':', alpha=0.5, which='minor')  # Minor grid lines

    if as_subplots==False:
        # Show the plot
        plt.show()

def drawGraph_population(seizures_df, limx=10, agetype='', as_subplots=False, thetype='all',output_file='output_file.csv'):
    if thetype == 'all':
        sub_df = seizures_df.copy()
    else:
        sub_df = seizures_df[seizures_df['type'] == thetype].copy()
    
    # Filter durations greater than 0 and convert to minutes
    filtered_sub = sub_df[sub_df['duration'] > 0].copy()
    #durations = filtered_sub['duration'] / 60

    if thetype == 'all':
        # Group by Unlinked_ID and type, then calculate median duration for each group
        patient_medians = filtered_sub.groupby(['Unlinked_ID'])['duration'].median().reset_index()
    else:
        # Group by Unlinked_ID and type, then calculate median duration for each group
        patient_medians = filtered_sub.groupby(['Unlinked_ID', 'type'])['duration'].median().reset_index()

    # Convert durations to minutes
    patient_medians['duration'] = patient_medians['duration'] / 60
    
    # Filter durations greater than 0
    durations = patient_medians[patient_medians['duration'] > 0]['duration']

    # Sort the durations and calculate the cumulative probabilities
    sorted_durations = np.sort(durations)
    cumulative_probs = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    
    ## Find the index of the duration closest to 5 minutes
    #index5 = np.abs(sorted_durations - 5).argmin()
    
    ## Get the cumulative probability at that index
    #cumulative_prob5 = cumulative_probs[index5]
    index90 = np.abs(cumulative_probs - 0.9).argmin()
    cumulative_prob90 = cumulative_probs[index90]
    prob90_time = sorted_durations[index90]

    # Create the plot
    if not as_subplots:
        plt.figure(figsize=(4, 6))
    

    if agetype == 'child':
        thecolor = 'blue'
    else:
        thecolor = 'red'

    plt.plot(sorted_durations, cumulative_probs * 100, label=agetype, color=thecolor)
    
    plt.axhline(y=90, linestyle='--', linewidth=2, color='gray')
    plt.axvline(x=prob90_time, linestyle='--', linewidth=2, color=thecolor)
    
    texty = 20 if thecolor == 'red' else 10
    haChoice = 'left' if thecolor == 'red' else 'right'
    textx = prob90_time + (-1 if thecolor == 'red' else 1) 
    if prob90_time > 10:
        textx += (-1 if thecolor == 'red' else 1)
    plt.text(textx, texty, f'{prob90_time:.0f}', ha=haChoice, va='center', color=thecolor)

    #plt.text(prob90_time, cumulative_prob90 * 100, f'{prob90_time}', ha='right', va='bottom')
    #print(f"Number of seizures: {len(durations)} for number of patients: {len(sub_df['Unlinked_ID'].unique())} {prob90_time} minutes for {agetype}")
    txt = f"pickone,{thetype},{agetype},{len(durations)},{len(filtered_sub['Unlinked_ID'].unique())},{prob90_time:.1f},{np.median(sorted_durations):.1f}"
    print(txt)
    # Open the file in append mode, which creates the file if it does not exist
    with open(output_file, 'a') as file:
        file.write(txt + '\n')

#    print(f"{agetype},{len(durations)},{len(sub_df['Unlinked_ID'].unique())},{prob90_time}")

    #plt.axvline(x=5, linestyle='--', linewidth=2)
    #plt.text(5, cumulative_prob5 * 100, f'{cumulative_prob5:.0%}', ha='right', va='bottom')
    #print(f'{cumulative_prob5} % at 5minutes for {agetype}')
    
    # Set the x-axis limit to limx minutes
    plt.xlim(0, limx)
    
    # Set the y-axis limit from 0 to 100%
    plt.ylim(0, 100)
    
    # Labels and title
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Cumulative Probability (%)')
    if not as_subplots:
        plt.title(f'CDF of Median Seizure Durations {agetype}')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set tick marks 
    #plt.xticks(np.arange(0, limx+1, limx/10))  # X-axis ticks from 0 to limx, every limx/10 minutes
    limx = np.max([limx, int(prob90_time*1.1)])
    
    if limx < 10:
        xstep = 1
    elif limx > 30:
        xstep = 10
    elif limx > 20:
        xstep = 5
    else:
        xstep = 2
    plt.xticks(np.arange(0,limx+1,xstep))
    plt.yticks(np.arange(0, 101, 10))  # Y-axis ticks from 0% to 100%, every 10%
    
    # Set minor tick marks
    ax = plt.gca()  # Get the current Axes instance
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for x-axis, between the major ticks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for y-axis, between the major ticks
    
    # Add minor grid lines
    ax.grid(True, linestyle=':', alpha=0.5, which='minor')  # Minor grid lines

    if not as_subplots:
        # Show the plot
        plt.show()
        
def process_unlinked_id(unlinked_id, group):
    valid_seizures = group[(group['duration'] > 0) & (group['duration'].notna())]
    
    if valid_seizures.empty:
        return unlinked_id, pd.Series(dtype=float)
    
    id_medians = valid_seizures.groupby('type')['duration'].median()
    
    if (id_medians < 1).any():
        print(f"Warning: Unlinked_ID {unlinked_id} has median duration < 1 second for types: {id_medians[id_medians < 1].index.tolist()}")
    
    return unlinked_id, id_medians

def analyze_seizure_durations(seizures_df):
    # Create a copy of the DataFrame to avoid modifying the original
    seizures_df = seizures_df.copy()
    
    # Remove rows where 'type' is NaN
    seizures_df = seizures_df.dropna(subset=['type'])

    # Ensure 'duration' is numeric and remove any NaN values
    seizures_df.loc[:, 'duration'] = pd.to_numeric(seizures_df['duration'], errors='coerce')
    seizures_df = seizures_df.dropna(subset=['duration'])

    # Get all seizure types
    all_types = seizures_df['type'].unique()

    # Group the dataframe by Unlinked_ID
    grouped = seizures_df.groupby('Unlinked_ID')

    # Process each group
    results = {}
    for unlinked_id, group in tqdm(grouped, desc="Processing Unlinked_IDs"):
        unlinked_id, id_medians = process_unlinked_id(unlinked_id, group)
        results[unlinked_id] = id_medians

    # Create DataFrame from results
    median_df = pd.DataFrame.from_dict(results, orient='index', columns=all_types)

    # Calculate statistics for each seizure type
    stats = []
    for seizure_type in all_types:
        type_data = median_df[seizure_type].dropna()
        
        if not type_data.empty:
            stats.append({
                'Type': seizure_type,
                'Min': type_data.min(),
                'Median': type_data.median(),
                'Max': type_data.max(),
                '95th': type_data.quantile(0.95),
                'Count': len(type_data)  # Number of Unlinked_IDs with this seizure type
            })

    # Create a DataFrame from the results
    result_df = pd.DataFrame(stats)

    # Sort the DataFrame by the median duration
    result_df = result_df.sort_values('Median', ascending=False)

    # Convert durations from seconds to minutes
    for col in ['Min', 'Median', 'Max', '95th']:
        result_df[col] = result_df[col] / 60

    # Round all numeric columns to 2 decimal places
    result_df = result_df.round(2)

    return result_df


def analyze_durations_all(seizures_df):
    # Create a copy of the DataFrame to avoid modifying the original
    seizures_df = seizures_df.copy()
    
    # Remove rows where 'type' is NaN
    seizures_df = seizures_df.dropna(subset=['type'])

    # Ensure 'duration' is numeric and remove any NaN values
    seizures_df.loc[:, 'duration'] = pd.to_numeric(seizures_df['duration'], errors='coerce')
    seizures_df = seizures_df.dropna(subset=['duration'])

    # Filter out durations <= 0
    seizures_df = seizures_df[seizures_df['duration'] > 0]

    # Get all seizure types
    all_types = seizures_df['type'].unique()

    # Calculate statistics for each seizure type
    stats = []
    for seizure_type in all_types:
        type_data = seizures_df[seizures_df['type'] == seizure_type]['duration']
        
        if not type_data.empty:
            stats.append({
                'Type': seizure_type,
                'Min': type_data.min(),
                'Median': type_data.median(),
                'Max': type_data.max(),
                '95th': type_data.quantile(0.95),
                'Count': len(type_data)  # Number of seizures of this type
            })

    # Create a DataFrame from the results
    result_df = pd.DataFrame(stats)

    # Sort the DataFrame by the median duration
    result_df = result_df.sort_values('Median', ascending=False)

    # Convert durations from seconds to minutes
    for col in ['Min', 'Median', 'Max', '95th']:
        result_df[col] = result_df[col] / 60

    # Round all numeric columns to 2 decimal places
    result_df = result_df.round(2)

    return result_df

def compare_results(res_by_id, res_all, ax1, ax2, agetype=''):
    # Merge the results
    merged = pd.merge(res_by_id, res_all, on='Type', suffixes=('_by_id', '_all'))
    
    # Sort by the 'Grouped by ID' median values
    merged = merged.sort_values('Median_by_id', ascending=True)

    y = np.arange(len(merged))
    height = 0.35

    # Plot median durations
    ax1.barh(y - height/2, merged['Median_by_id'], height, label='Grouped by ID', color='lightblue')
    ax1.barh(y + height/2, merged['Median_all'], height, label='All Seizures', color='blue')

    ax1.set_yticks(y)
    ax1.set_yticklabels(merged['Type'])
    ax1.invert_yaxis()  # labels read top-to-bottom
    
    ax1.set_xlabel('Median Duration (minutes)')
    #ax1.set_ylabel('Seizure Type')
    ax1.set_title(f'{agetype} Seizures: Comparison of Median Durations')
    ax1.set_xlim([0,8])
    ax1.legend()

    # Add gridlines
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add value labels for median
    for i, (v1, v2) in enumerate(zip(merged['Median_by_id'], merged['Median_all'])):
        ax1.text(v1, i - height/2, f'{v1:.1f}', va='center', ha='left', fontweight='bold')
        ax1.text(v2, i + height/2, f'{v2:.1f}', va='center', ha='left', fontweight='bold')

    # Plot 95th percentile durations
    ax2.barh(y - height/2, merged['95th_by_id'], height, label='Grouped by ID', color='pink')
    ax2.barh(y + height/2, merged['95th_all'], height, label='All Seizures', color='red')

    ax2.set_yticks(y)
    ax2.set_yticklabels(merged['Type'])
    ax2.invert_yaxis()  # labels read top-to-bottom
    
    ax2.set_xlabel('95th Percentile Duration (minutes)')
    #ax2.set_ylabel('Seizure Type')
    ax2.set_title(f'{agetype} Seizures: Comparison of 95th Percentile Durations')
    ax2.legend()

    # Add gridlines
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add value labels for 95th percentile
    for i, (v1, v2) in enumerate(zip(merged['95th_by_id'], merged['95th_all'])):
        ax2.text(v1, i - height/2, f'{v1:.1f}', va='center', ha='left', fontweight='bold')
        ax2.text(v2, i + height/2, f'{v2:.1f}', va='center', ha='left', fontweight='bold')

    ax2.set_xlim([0,300])
    



def make_boxplot(child_seizures, adult_seizures):
    # Prepare data for children
    child_df = prepare_data_for_boxplot(child_seizures, 'Child')
    
    # Prepare data for adults
    adult_df = prepare_data_for_boxplot(adult_seizures, 'Adult')
    
    # Combine the dataframes
    combined_df = pd.concat([child_df, adult_df])
    
    # Calculate summary statistics
    summary = combined_df.groupby(['type', 'age_group'])['duration'].agg(['count', 'median']).reset_index()
    summary = summary.sort_values(by=['age_group', 'type'], ascending=[False, True])
        
    # Create the boxplot
    ax = sns.boxplot(y='type', x='duration', hue='age_group', data=combined_df, 
                     orient='h', showfliers=False, palette=['blue', 'red'])
    
    plt.title('One per patient median durations by type: Children vs Adults')
    plt.xlabel('Duration (minutes)')
    
    # Add count and median as text
    for i, (_, row) in enumerate(summary.iterrows()):
        y_pos = i - 0.2 if row['age_group'] == 'Child' else i + 0.2
    #    plt.text(plt.xlim()[1], y_pos, 
    #             f"{row['age_group'][0]}:n={int(row['count'])}, m={row['median']:.2f}", 
    #             verticalalignment='center', horizontalalignment='left', fontsize=8)
    
    # Add sub-tic marks and sub-grid lines
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.grid(True, axis='x', which='both', linestyle='--', alpha=0.7)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    

def prepare_data_for_boxplot(seizures_df, age_group):
    df = seizures_df.copy()
    df = df.dropna(subset=['type'])
    df.loc[:, 'duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] > 0]
    df.loc[:, 'duration'] = df['duration'] / 60
    medians = df.groupby(['Unlinked_ID', 'type'])['duration'].median().reset_index()
    medians['age_group'] = age_group
    return medians


def make_table_1(seizures_df, profiles_df, csv_file):
    # Get unique Unlinked_ID values from seizures dataframe
    unique_ids = seizures_df['Unlinked_ID'].unique()
    
    # Filter profiles dataframe to include only patients in the seizures dataframe
    relevant_profiles = profiles_df[profiles_df['Unlinked_ID'].isin(unique_ids)].copy()
    
    # Calculate total number of profiles
    total_profiles = len(relevant_profiles)
    
    # Calculate age
    seizures_df['Date_Time'] = pd.to_datetime(seizures_df['Date_Time'], errors='coerce')
    median_seizure_dates = seizures_df.groupby('Unlinked_ID')['Date_Time'].median()
    
    # Handle invalid birth dates
    relevant_profiles['Birth_Date'] = pd.to_datetime(relevant_profiles['Birth_Date'], errors='coerce')
    
    # Calculate age, handling NaT values and excluding negative ages
    relevant_profiles['Median_Seizure_Date'] = relevant_profiles['Unlinked_ID'].map(median_seizure_dates)
    relevant_profiles['Age'] = (relevant_profiles['Median_Seizure_Date'] - relevant_profiles['Birth_Date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Exclude negative ages
    relevant_profiles = relevant_profiles[relevant_profiles['Age'] >= 0]
    
    # Recalculate total number of profiles after age exclusion
    total_profiles_after_age_filter = len(relevant_profiles)
    
    # Summarize age
    age_summary = relevant_profiles['Age'].agg(['count', 'mean', 'std', 'min', 'max'])
    
    # Summarize gender
    gender_summary = relevant_profiles['Gender'].value_counts()
    gender_percentages = (gender_summary / total_profiles_after_age_filter * 100).round(1)
    
    # List of medical conditions to count
    conditions = [
        'Congenital_Condition', 'Brain_Tumors', 'Brain_Trauma', 'Brain_Trauma_Hematomas',
        'Infectious_Diseases', 'Alzheimers', 'Stroke', 'Heart_Attack', 'Brain_Surgery',
        'Metabolic_Disorder', 'Genetic_Abnormalities', 'Electrolyte_Disturbances',
        'Brain_Injury_During_Fetal_Development', 'Lack_Of_Oxygen_During_Birth',
        'Maternal_Drug_Or_Alcohol_Abuse', 'Brain_Malformations', 'Lead_Exposure',
        'Carbon_Monoxide_Exposure', 'Alcohol_Or_Drug_Abuse', 'High_Fever'
    ]
    
    # Count medical conditions (only for columns that exist)
    existing_conditions = [col for col in conditions if col in relevant_profiles.columns]
    condition_counts = relevant_profiles[existing_conditions].notna().sum()
    condition_percentages = (condition_counts / total_profiles_after_age_filter * 100).round(1)
    
    # Prepare data for summary DataFrame
    data = {
        'Characteristic': ['N (total profiles before age filter)', 'N (total profiles after age filter)', 'Age count', 'Age mean (years)', 'Age std (years)', 'Age min (years)', 'Age max (years)'] +
                          [f'Gender: {gender}' for gender in gender_summary.index] +
                          [f'Condition: {condition}' for condition in conditions],
        'Value': [
            f"{total_profiles:.0f}",
            f"{total_profiles_after_age_filter:.0f}",
            f"{age_summary['count']:.0f}",
            f"{age_summary['mean']:.2f}",
            f"{age_summary['std']:.2f}",
            f"{age_summary['min']:.2f}",
            f"{age_summary['max']:.2f}"
        ] +
        [f"{count:.0f} ({percentage:.1f}%)" for count, percentage in zip(gender_summary.values, gender_percentages.values)] +
        [f"{condition_counts.get(condition, 0):.0f} ({condition_percentages.get(condition, 0):.1f}%)" for condition in conditions]
    }
    
    # Create summary DataFrame
    summary = pd.DataFrame(data)
    summary.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    return summary

def bigCDF(seizures_df, max_duration=None,agetype='',as_subplots=False):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = seizures_df.copy()
    
    # Ensure 'duration' is in minutes
    df['duration_minutes'] = df['duration'] / 60
    
    if max_duration:
        # Filter out durations greater than max_duration
        df = df[df['duration_minutes'] <= max_duration]
    
    # Filter out durations <1 second
    df = df[df['duration_minutes'] > 1/60]
    
    # Replace 'nan' with 'Unknown*' in the 'type' column
    df['type'] = df['type'].fillna('Unknown*')
    
    # Calculate 90th percentile duration for each seizure type
    percentile_90 = df.groupby('type')['duration_minutes'].quantile(0.9).sort_values()
    
    # Get sorted unique seizure types
    seizure_types = percentile_90.index.tolist()
    
    if as_subplots==False:
        # Set up the plot
        plt.figure(figsize=(5, 12))

    # Color map (red to blue)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(seizure_types)))
    
    # Markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    # Plot CDF for each seizure type
    for i, seizure_type in enumerate(seizure_types):
        type_data = df[df['type'] == seizure_type]['duration_minutes']
        
        # Sort the durations and calculate the cumulative probabilities
        sorted_durations = np.sort(type_data)
        cumulative_probs = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        
        # Plot the CDF
        plt.plot(sorted_durations, cumulative_probs * 100, 
                 color=colors[i], marker=markers[i % len(markers)], 
                 linestyle='-', markersize=4, markevery=0.1, 
                 label=f"{seizure_type} (90%ile: {percentile_90[seizure_type]:.2f})")
    
    # Customize the plot
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Cumulative Probability (%)')
    plt.title(f'CDF Durations by Type, {agetype}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 30)
    plt.ylim(0, 100)
    
    # Add legend
    plt.legend(title='Seizure Type', loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(title='Seizure Type', loc='lower right')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    
    if as_subplots==False:
        # Show the plot
        plt.show()


def makeBoxPlotsV2(seizures_child, seizures_adult):
    # Prepare data for children
    child_df = prepare_data_for_plot(seizures_child, 'Child')
    
    # Prepare data for adults
    adult_df = prepare_data_for_plot(seizures_adult, 'Adult')
    
    # Combine the dataframes
    combined_df = pd.concat([child_df, adult_df])
    
    # Calculate summary statistics
    summary = combined_df.groupby(['type', 'age_group'])['duration'].agg(['count', 'median']).reset_index()
    summary = summary.sort_values(by=['type', 'age_group'], ascending=[True, False])
    
    # Set up the plot
    plt.figure(figsize=(12, 15))
    
    # Create the boxplot
    ax = sns.boxplot(y='type', x='duration', hue='age_group', data=combined_df, 
                     orient='h', width=0.8, fliersize=2,
                     palette={'Child': 'blue', 'Adult': 'red'})
    
    plt.title('Seizure Durations by Type: Children vs Adults', fontsize=16)
    plt.ylabel('Seizure Type', fontsize=12)
    plt.xlabel('Duration (minutes)', fontsize=12)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Add count and median as text
    for i, (_, row) in enumerate(summary.iterrows()):
        x_pos = ax.get_xlim()[1]
        y_pos = i + (0.2 if row['age_group'] == 'Child' else -0.2)
        plt.text(x_pos, y_pos, f"{row['age_group'][0]}:n={int(row['count'])}, m={row['median']:.1f}", 
                 verticalalignment='center', horizontalalignment='left', fontsize=8)
    
    # Adjust layout and display the plot
    #plt.tight_layout()
    plt.savefig('Fig4-median_simpler-boxplot.png',dpi=300)
    plt.show()

def prepare_data_for_plot(seizures_df, age_group):
    df = seizures_df.copy()
    df = df.dropna(subset=['type'])
    df.loc[:, 'duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] > 0]
    df.loc[:, 'duration'] = df['duration'] / 60  # Convert to minutes
    df['age_group'] = age_group
    return df

def drawSimplerBoxes(res_child, res_adult):
    print(res_child.head())
    print(res_adult.head())

    # Prepare data
    child_data = prepare_data_for_simplerboxplot(res_child, 'Child')
    adult_data = prepare_data_for_simplerboxplot(res_adult, 'Adult')
    combined_data = pd.concat([child_data, adult_data])

    # Identify the column representing the central tendency
    numeric_columns = child_data.select_dtypes(include=[np.number]).columns
    central_tendency_col = next((col for col in numeric_columns if col not in ['Count']), None)
    
    if central_tendency_col is None:
        raise ValueError("Could not find a suitable numeric column for central tendency")

    # Sort data by adult central tendency
    type_order = adult_data.groupby('Type')[central_tendency_col].median().sort_values(ascending=True).index

    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Create the boxplot
    ax = sns.boxplot(y='Type', x=central_tendency_col, hue='Age Group', data=combined_data, 
                     orient='h', order=type_order, palette={'Child': 'lightblue', 'Adult': 'pink'})

    # Customize the plot
    plt.title(f'{central_tendency_col} Seizure Durations by Type: Children vs Adults', fontsize=16)
    plt.xlabel(f'{central_tendency_col} Duration (minutes)', fontsize=12)
    plt.ylabel('Seizure Type', fontsize=12)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Add count information
    for i, seizure_type in enumerate(type_order):
        child_count = res_child[res_child['Type'] == seizure_type]['Count'].values[0]
        adult_count = res_adult[res_adult['Type'] == seizure_type]['Count'].values[0]
        
        plt.text(ax.get_xlim()[1], i, f'C:{child_count}, A:{adult_count}', 
                 verticalalignment='center', horizontalalignment='left', fontsize=8)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def prepare_data_for_simplerboxplot(res_df, age_group):
    # Identify columns representing statistics (excluding 'Type' and 'Count')
    stat_columns = [col for col in res_df.columns if col not in ['Type', 'Count']]

    # Melt the dataframe to create a format suitable for boxplot
    melted_df = pd.melt(res_df, id_vars=['Type', 'Count'], 
                        value_vars=stat_columns, 
                        var_name='Statistic', value_name='Value')
    
    # Add age group information
    melted_df['Age Group'] = age_group
    
    return melted_df

def make_a_pair_of_CDFs(child_seizures, adult_seizures, thetype='all',titlestr='All types',fname='Fig1-CDF-all.png',output_file='CDF_data.csv',xlimlist=(10,18)):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    drawGraph(child_seizures,agetype='child',as_subplots=True,thetype=thetype,output_file=output_file)
    #plt.subplot(1,2,2)
    drawGraph(adult_seizures,agetype='adult',as_subplots=True,thetype=thetype,output_file=output_file)
    plt.title(titlestr)
    plt.legend()
    #plt.savefig('Fig1-CDF-all.png',dpi=300)
    #plt.show()

    #  first
    #plt.figure(figsize=(3,3))
    plt.subplot(1,2,2)
    drawGraph_population(child_seizures,agetype='child',as_subplots=True,limx=xlimlist[0],thetype=thetype,output_file=output_file)
    #plt.subplot(1,2,2)
    drawGraph_population(adult_seizures,agetype='adult',as_subplots=True,limx=xlimlist[1],thetype=thetype,output_file=output_file)
    plt.ylabel('') # remove xlabel
    #plt.yticks([])  # Remove y-axis tick labels

    plt.title(f'(1 per patient)')
    #plt.legend()
    #plt.savefig('Fig1b-CDF-pop.png',dpi=300)
    plt.tight_layout()
    plt.savefig(fname,dpi=300)
    plt.show()

def do_full_display(do_raw_read=False):

    # Read the SeizureTracker CSV file directly, or just a pickle of it already read in
    if do_raw_read:
        #profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20220808_A.csv')
        profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20240717.csv')

        print('Saving pickles...',end='')
        profiles.to_pickle('profiles.pkl')
        seizures.to_pickle('seizures.pkl')
        print('done.')
    else:
        print('Loading pickles...',end='')
        profiles = pd.read_pickle('profiles.pkl')
        seizures = pd.read_pickle('seizures.pkl')
        print('done.')

    # Replace 'nan' with 'Unknown*' in the 'type' column
    seizures['type'] = seizures['type'].fillna('Unknown*')
    # Replace 'Desconocido' with 'Unknown*' in the 'type' column
    seizures['type'] = seizures['type'].replace('Desconocido', 'Unknown*')


    csv_child = 'table1_child.csv'
    csv_adult = 'table1_adult.csv'
    output_file = 'CDF_data.csv'

    child_seizures, adult_seizures = age_cut(seizures, profiles)
    # child seizures first
    make_table_1(child_seizures, profiles,csv_child)
    make_table_1(adult_seizures, profiles,csv_adult)

    
    #  first

    txtcols = 'cdftype,type,agetype,nseizures,npatients,prob90_time,median'
    with open(output_file, 'w') as file:
        file.write(txtcols + '\n')

    # Create child_seizures_GTC with GTC combined types
    child_seizures_GTC = child_seizures.copy()
    child_seizures_GTC['type'] = child_seizures_GTC['type'].replace({
        'Focal to bilateral tonic clonic*': 'GTC',
        'Tonic Clonic':                    'GTC'
    })
    # Create adult_seizures_GTC with GTC combined types
    adult_seizures_GTC = adult_seizures.copy()
    adult_seizures_GTC['type'] = adult_seizures_GTC['type'].replace({
        'Focal to bilateral tonic clonic*': 'GTC',
        'Tonic Clonic':                    'GTC'
    })

    # Create child_seizures_IA with all unaware types
    #IA = 'Impaired aware'
    IA = 'Impaired Consciousness'
    child_seizures_IA = child_seizures.copy()
    child_seizures_IA['type'] = child_seizures_IA['type'].replace({
        'Absence': IA,
        'Atypical Absence': IA,
        'Focal to bilateral tonic clonic*': IA,
        'Focal impaired awareness*': IA,
        'Tonic': IA,
        'Clonic': IA,
        'Gelastic': IA,
        'Tonic Clonic': IA
    })
    # Create adult_seizures_GTC with GTC combined types
    adult_seizures_IA = adult_seizures.copy()
    adult_seizures_IA['type'] = adult_seizures_GTC['type'].replace({
        'Absence': IA,
        'Atypical Absence': IA,
        'Focal to bilateral tonic clonic*': IA,
        'Focal impaired awareness*': IA,
        'Tonic': IA,
        'Clonic': IA,
        'Gelastic': IA,
        'Tonic Clonic': IA
    })
    
    # build a special FIGURE 1 with 4 parts
    fname = 'Figure1-comboFig.png'
    output_file = 'Figure1-combofile.csv'
    plt.figure(figsize=(6,6))
    plt.subplot(2,2,1)
    thetype = 'all'
    drawGraph(child_seizures,agetype='child',as_subplots=True,thetype=thetype,output_file=output_file)
    drawGraph(adult_seizures,agetype='adult',as_subplots=True,thetype=thetype,output_file=output_file)
    plt.xlim(0,20)
    ax = plt.gca()  # Get the current Axes instance
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for x-axis, between the major ticks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for y-axis, between the major ticks
    # Add minor grid lines
    ax.grid(True, linestyle=':', alpha=0.5, which='minor')  # Minor grid lines
    plt.xticks([])  # Remove x-axis tick labels
    plt.ylabel('') # remove ylabel
    plt.legend()
    plt.title('All by seizures')
    plt.xlabel('') # remove xlabel
    plt.subplot(2,2,2)
    thetype = 'all'
    xlimlist=(10,20)
    drawGraph_population(child_seizures,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph_population(adult_seizures,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    plt.title('All by patient')
    plt.xlabel('') # remove xlabel
    plt.ylabel('') # remove ylabel
    plt.xticks([])  # Remove x-axis tick labels
    plt.subplot(2,2,3)
    thetype = 'GTC'
    drawGraph_population(child_seizures_GTC,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph_population(adult_seizures_GTC,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    plt.title('GTC+FBTC by patient')
    plt.subplot(2,2,4)
    thetype = IA
    drawGraph_population(child_seizures_IA,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph_population(adult_seizures_IA,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    #plt.title('Focal impaired consciousness by patient')
    plt.title('Focal impaired\nconsciousness by patient')
    plt.ylabel('') # remove ylabel
    plt.xlabel('') # remove xlabel
    plt.xticks([])  # Remove x-axis tick labels
    plt.tight_layout()
    plt.savefig(fname,dpi=300)
    plt.show()

    # this is mainly just for the output file.
    thetype = 'GTC'
    drawGraph(child_seizures_GTC,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph(adult_seizures_GTC,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    plt.show()
    thetype = IA
    drawGraph(child_seizures_IA,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph(adult_seizures_IA,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    plt.show()


    make_a_pair_of_CDFs(child_seizures, adult_seizures, thetype='all',titlestr='All types',fname='Fig1-CDF-all.png',output_file=output_file,xlimlist=(10,25))

   
    make_a_pair_of_CDFs(child_seizures_GTC, adult_seizures_GTC, thetype='GTC',titlestr='GTC + FBTC',fname='Fig1C-CDF-GTC.png',output_file=output_file,xlimlist=(10,25))


    for thetype in seizures['type'].unique():
        print(f'Processing {thetype}...')
        make_a_pair_of_CDFs(child_seizures, adult_seizures, thetype=thetype,titlestr=thetype,fname=f'Fig1-CDF-{thetype}.png',output_file=output_file)


    # Create a figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Child seizures
    res_child = analyze_seizure_durations(child_seizures)
    res_all_child = analyze_durations_all(child_seizures)
    compare_results(res_child, res_all_child, ax1, ax2, agetype='Child')

    # Adult seizures
    res_adult = analyze_seizure_durations(adult_seizures)
    res_all_adult = analyze_durations_all(adult_seizures)
    compare_results(res_adult, res_all_adult, ax3, ax4, agetype='Adult')

    plt.tight_layout()
    plt.savefig('APPENDOX-fig1-compare.png',dpi=300)
    plt.show()



    # make boxplots third
    plt.figure(figsize=(12, 7))
    make_boxplot(child_seizures, adult_seizures)
    plt.savefig('Fig2-boxplot.png',dpi=300)
    plt.show()

    if 0:
        drawSimplerBoxes(res_child, res_adult)
        makeBoxPlotsV2(child_seizures, adult_seizures)
    #plt.figure(figsize=(7, 10))
    #plt.subplot(2,1,1)
    #bigCDF(child_seizures, max_duration=None,agetype='child',as_subplots=True)
    #plt.subplot(2,1,2)
    #bigCDF(adult_seizures, max_duration=None,agetype='adult',as_subplots=True)
    #plt.show()


# Usage
#file_path = 'path/to/your/seizuretracker_file.csv'
#profiles, seizures = read_seizuretracker_csv(file_path)

#print("Profiles DataFrame:")
#print(profiles.head())
#print("\nSeizures DataFrame:")
#print(seizures.head())