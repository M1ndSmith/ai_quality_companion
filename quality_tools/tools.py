import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Wedge, Polygon
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch
from datetime import datetime
import math
from typing_extensions import Dict, List
import os



def create_sipoc_diagram(process_name, suppliers, inputs, process_steps, outputs, customers):
    """
    Creates a visual SIPOC diagram using Matplotlib.
    
    Parameters:
    - process_name (str): Name of the process being mapped
    - suppliers (list): List of suppliers/vendors for the process
    - inputs (list): List of inputs required by the process
    - process_steps (list): List of key steps in the process (3-7 high-level steps)
    - outputs (list): List of outputs produced by the process
    - customers (list): List of customers/stakeholders who receive the outputs
    
    Returns:
    - dict: Contains the figure object and a message
    """
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'SIPOC Diagram: {process_name}', fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1.5, 1, 1])
    
    # Define colors for each section
    colors = {
        'suppliers': '#FFDDC1',  # Light orange
        'inputs': '#C1FFD7',     # Light green
        'process': '#C1D3FF',    # Light blue
        'outputs': '#FFC1F3',    # Light pink
        'customers': '#FFFAC1'   # Light yellow
    }
    
    # Create axes for each SIPOC component
    ax_suppliers = fig.add_subplot(gs[0, 0])
    ax_inputs = fig.add_subplot(gs[0, 1])
    ax_process = fig.add_subplot(gs[0, 2])
    ax_outputs = fig.add_subplot(gs[0, 3])
    ax_customers = fig.add_subplot(gs[0, 4])
    
    # Function to create a component box
    def create_component_box(ax, title, items, color):
        ax.set_title(title, pad=10, fontweight='bold')
        ax.set_facecolor(color)
        ax.axis('off')
        
        # Add items to the box
        for i, item in enumerate(items):
            ax.text(0.5, 0.9 - (i * 0.15), item, 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Create each component box
    create_component_box(ax_suppliers, "Suppliers", suppliers, colors['suppliers'])
    create_component_box(ax_inputs, "Inputs", inputs, colors['inputs'])
    create_component_box(ax_process, "Process", process_steps, colors['process'])
    create_component_box(ax_outputs, "Outputs", outputs, colors['outputs'])
    create_component_box(ax_customers, "Customers", customers, colors['customers'])
    
    # Add arrows between components
    fig.text(0.18, 0.5, "→", fontsize=20, ha='center', va='center')
    fig.text(0.38, 0.5, "→", fontsize=20, ha='center', va='center')
    fig.text(0.62, 0.5, "→", fontsize=20, ha='center', va='center')
    fig.text(0.82, 0.5, "→", fontsize=20, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the diagram
    output_path = 'sipoc_diagram.png'
    fig.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory
    
    # Create and save CSV data
    # First, create a dictionary with all SIPOC components
    sipoc_data = {
        'Process Name': [process_name] * max(len(suppliers), len(inputs), len(process_steps), len(outputs), len(customers)),
        'Suppliers': suppliers + [''] * (max(len(suppliers), len(inputs), len(process_steps), len(outputs), len(customers)) - len(suppliers)),
        'Inputs': inputs + [''] * (max(len(suppliers), len(inputs), len(process_steps), len(outputs), len(customers)) - len(inputs)),
        'Process Steps': process_steps + [''] * (max(len(suppliers), len(inputs), len(process_steps), len(outputs), len(customers)) - len(process_steps)),
        'Outputs': outputs + [''] * (max(len(suppliers), len(inputs), len(process_steps), len(outputs), len(customers)) - len(outputs)),
        'Customers': customers + [''] * (max(len(suppliers), len(inputs), len(process_steps), len(outputs), len(customers)) - len(customers))
    }
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(sipoc_data)
    df.to_csv('sipoc_data.csv', index=False)
    
    return {
        'message': f"SIPOC diagram and data have been saved to:\n" +
                  f"- {output_path}\n" +
                  f"- sipoc_data.csv"
    }






def perform_fmea_analysis(failure_modes, effects, causes, current_controls, 
                          severity_scores, occurrence_scores, detection_scores,
                          action_threshold=100):
    """
    Perform Failure Mode and Effects Analysis (FMEA) with visualization and recommendations.
    
    Parameters:
    - failure_modes (list): List of potential failure modes
    - effects (list): List of effects for each failure mode
    - causes (list): List of causes for each failure mode
    - current_controls (list): List of current controls for each failure mode
    - severity_scores (list): Severity scores (1-10) for each failure mode
    - occurrence_scores (list): Occurrence scores (1-10) for each failure mode
    - detection_scores (list): Detection scores (1-10) for each failure mode
    - action_threshold (int): RPN threshold for recommending actions (default=100)
    
    Returns:
    - dict: Contains FMEA table (DataFrame), risk matrix plot (Figure), and recommendations
    """
    
    # Convert nested lists to simple lists if needed
    if isinstance(severity_scores[0], list):
        severity_scores = [max(scores) for scores in severity_scores]
    if isinstance(occurrence_scores[0], list):
        occurrence_scores = [max(scores) for scores in occurrence_scores]
    if isinstance(detection_scores[0], list):
        detection_scores = [max(scores) for scores in detection_scores]
    
    # Convert dictionary lists to simple lists if needed
    if isinstance(causes[0], dict):
        causes = [', '.join(list(cause.values())[0]) for cause in causes]
    if isinstance(current_controls[0], dict):
        current_controls = [', '.join(list(control.values())[0]) for control in current_controls]
    if isinstance(effects[0], list):
        effects = [', '.join(effect) for effect in effects]
    
    # Validate input lengths
    if not all(len(lst) == len(failure_modes) for lst in 
              [effects, causes, current_controls, severity_scores, 
               occurrence_scores, detection_scores]):
        raise ValueError("All input lists must have the same length")
    
    # Calculate Risk Priority Numbers (RPN)
    rpn_scores = [s * o * d for s, o, d in 
                 zip(severity_scores, occurrence_scores, detection_scores)]
    
    # Create FMEA table
    fmea_data = {
        'Failure Mode': failure_modes,
        'Effect': effects,
        'Cause': causes,
        'Current Controls': current_controls,
        'Severity (S)': severity_scores,
        'Occurrence (O)': occurrence_scores,
        'Detection (D)': detection_scores,
        'RPN (S×O×D)': rpn_scores
    }
    
    fmea_df = pd.DataFrame(fmea_data)
    fmea_df['Priority'] = fmea_df['RPN (S×O×D)'].rank(ascending=False, method='min')
    
    # Save FMEA table to CSV
    os.makedirs('output', exist_ok=True)
    fmea_df.to_csv('output/fmea_table.csv', index=False)
    
    # Generate risk matrix visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create risk matrix background
    for severity in range(1, 11):
        for occurrence in range(1, 11):
            risk_level = severity * occurrence
            color_intensity = min(0.1 + (risk_level / 100) * 0.9, 1)
            
            if risk_level >= 64:
                color = (1, 0.5 - color_intensity/2, 0.5 - color_intensity/2)  # Reds
            elif risk_level >= 16:
                color = (1, 1, 0.5 - color_intensity/3)  # Yellows
            else:
                color = (0.5 + color_intensity/2, 1, 0.5 + color_intensity/2)  # Greens
                
            ax.add_patch(Rectangle((occurrence-0.5, severity-0.5), 1, 1, 
                                 facecolor=color, edgecolor='white'))
    
    # Plot each failure mode
    for i, row in fmea_df.iterrows():
        ax.scatter(row['Occurrence (O)'], row['Severity (S)'], 
                  s=row['RPN (S×O×D)']*5, alpha=0.7,
                  label=f"{i+1}. {row['Failure Mode']} (RPN: {row['RPN (S×O×D)']})")
    
    # Matrix formatting
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_xlabel('Occurrence (O)', fontweight='bold')
    ax.set_ylabel('Severity (S)', fontweight='bold')
    ax.set_title('FMEA Risk Matrix', fontweight='bold', pad=20)
    
    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             title='Failure Modes with RPN')
    
    plt.tight_layout()
    
    # Save risk matrix
    fig.savefig('output/fmea_risk_matrix.png', bbox_inches='tight')
    plt.close(fig)
    
    # Generate recommendations
    high_risk = fmea_df[fmea_df['RPN (S×O×D)'] >= action_threshold]
    recommendations = []
    
    for _, row in high_risk.iterrows():
        rec = {
            'Failure Mode': row['Failure Mode'],
            'RPN': row['RPN (S×O×D)'],
            'Recommended Actions': []
        }
        
        # Action recommendations based on scores
        if row['Severity (S)'] >= 8:
            rec['Recommended Actions'].append("Implement design changes to reduce severity")
        if row['Occurrence (O)'] >= 8:
            rec['Recommended Actions'].append("Improve process controls to reduce frequency")
        if row['Detection (D)'] >= 8:
            rec['Recommended Actions'].append("Enhance detection methods with additional testing/inspection")
        
        if not rec['Recommended Actions']:
            rec['Recommended Actions'].append("Monitor closely and consider preventive measures")
        
        recommendations.append(rec)
    
    # Save recommendations to text file
    with open('output/fmea_recommendations.txt', 'w') as f:
        f.write("FMEA Recommendations\n")
        f.write("===================\n\n")
        for rec in recommendations:
            f.write(f"Failure Mode: {rec['Failure Mode']}\n")
            f.write(f"RPN: {rec['RPN']}\n")
            f.write("Recommended Actions:\n")
            for action in rec['Recommended Actions']:
                f.write(f"- {action}\n")
            f.write("\n")
    
    return {
        'message': "FMEA analysis completed. Results saved to:\n" +
                  "- output/fmea_table.csv\n" +
                  "- output/fmea_risk_matrix.png\n" +
                  "- output/fmea_recommendations.txt"
    }







def generate_5s_report(areas, sort_scores, set_order_scores, shine_scores, 
                      standardize_scores, sustain_scores, before_images=None, 
                      after_images=None, notes=None):
    """
    Generate a 5S report with scoring, visualization, and improvement tracking.
    
    Parameters:
    - areas (list): List of work areas being evaluated
    - sort_scores (list): Scores (1-5) for Sort (Seiri) step
    - set_order_scores (list): Scores (1-5) for Set in Order (Seiton) step
    - shine_scores (list): Scores (1-5) for Shine (Seiso) step
    - standardize_scores (list): Scores (1-5) for Standardize (Seiketsu) step
    - sustain_scores (list): Scores (1-5) for Sustain (Shitsuke) step
    - before_images (list): Optional list of before image paths
    - after_images (list): Optional list of after image paths
    - notes (list): Optional list of notes for each area
    
    Returns:
    - dict: Contains report DataFrame, radar chart, and improvement recommendations
    """
    
    # If multiple scores are provided for a single area, take the average
    if len(areas) == 1 and len(sort_scores) > 1:
        sort_scores = [sum(sort_scores) / len(sort_scores)]
        set_order_scores = [sum(set_order_scores) / len(set_order_scores)]
        shine_scores = [sum(shine_scores) / len(shine_scores)]
        standardize_scores = [sum(standardize_scores) / len(standardize_scores)]
        sustain_scores = [sum(sustain_scores) / len(sustain_scores)]
    
    # Validate inputs
    if not all(len(lst) == len(areas) for lst in [sort_scores, set_order_scores, 
                                                shine_scores, standardize_scores, 
                                                sustain_scores]):
        raise ValueError("All score lists must match length of areas list")
    
    # Create scoring dataframe
    data = {
        'Area': areas,
        '1. Sort (Seiri)': sort_scores,
        '2. Set in Order (Seiton)': set_order_scores,
        '3. Shine (Seiso)': shine_scores,
        '4. Standardize (Seiketsu)': standardize_scores,
        '5. Sustain (Shitsuke)': sustain_scores
    }
    
    df = pd.DataFrame(data)
    df['Total Score'] = df.iloc[:, 1:].sum(axis=1)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save the report to CSV
    df.to_csv('output/5s_report.csv', index=False)
    
    # Generate radar chart
    fig_radar, ax_radar = plt.subplots(figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the loop
    
    # Plot each area
    for i, area in enumerate(areas):
        scores = [sort_scores[i], set_order_scores[i], shine_scores[i], 
                 standardize_scores[i], sustain_scores[i]]
        scores = np.concatenate((scores, [scores[0]]))  # close the loop
        ax_radar.plot(angles, scores, 'o-', linewidth=2, label=area)
        ax_radar.fill(angles, scores, alpha=0.25)
    
    # Set radar chart properties
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(['Sort', 'Set in Order', 'Shine', 'Standardize', 'Sustain'])
    ax_radar.set_ylim(0, 5)
    ax_radar.set_yticks(range(1, 6))
    ax_radar.set_title('5S Assessment Radar Chart', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save radar chart
    fig_radar.savefig('output/5s_radar_chart.png', bbox_inches='tight')
    plt.close(fig_radar)
    
    # Generate bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    
    # Set up bar chart data
    x = np.arange(len(areas))
    width = 0.15
    
    # Plot bars for each 5S category
    ax_bar.bar(x - 2*width, sort_scores, width, label='Sort')
    ax_bar.bar(x - width, set_order_scores, width, label='Set in Order')
    ax_bar.bar(x, shine_scores, width, label='Shine')
    ax_bar.bar(x + width, standardize_scores, width, label='Standardize')
    ax_bar.bar(x + 2*width, sustain_scores, width, label='Sustain')
    
    # Set bar chart properties
    ax_bar.set_ylabel('Score')
    ax_bar.set_title('5S Assessment Scores by Area')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(areas)
    ax_bar.set_ylim(0, 5)
    ax_bar.legend()
    
    # Save bar chart
    fig_bar.savefig('output/5s_bar_chart.png', bbox_inches='tight')
    plt.close(fig_bar)
    
    # Generate recommendations
    recommendations = []
    for i, area in enumerate(areas):
        area_recs = []
        scores = [sort_scores[i], set_order_scores[i], shine_scores[i], 
                 standardize_scores[i], sustain_scores[i]]
        
        if min(scores) < 3:
            area_recs.append(f"Focus on improving {['Sort', 'Set in Order', 'Shine', 'Standardize', 'Sustain'][scores.index(min(scores))]} (score: {min(scores):.1f})")
        if max(scores) >= 4:
            area_recs.append(f"Maintain high standards in {['Sort', 'Set in Order', 'Shine', 'Standardize', 'Sustain'][scores.index(max(scores))]} (score: {max(scores):.1f})")
        if not area_recs:
            area_recs.append("Maintain current 5S standards - good job!")
            
        recommendations.append({
            'Area': area,
            'Recommendations': area_recs
        })
    
    # Save recommendations to text file
    with open('output/5s_recommendations.txt', 'w') as f:
        f.write("5S Assessment Recommendations\n")
        f.write("===========================\n\n")
        for rec in recommendations:
            f.write(f"Area: {rec['Area']}\n")
            f.write("Recommendations:\n")
            for action in rec['Recommendations']:
                f.write(f"- {action}\n")
            f.write("\n")
    
    return {
        'message': "5S assessment completed. Results saved to:\n" +
                  "- output/5s_report.csv\n" +
                  "- output/5s_radar_chart.png\n" +
                  "- output/5s_bar_chart.png\n" +
                  "- output/5s_recommendations.txt"
    }



def five_whys_analysis(problem_statement, whys_list, countermeasures=None):
    """
    Performs a 5 Whys analysis and creates a visual representation.
    
    Parameters:
    - problem_statement (str): The initial problem to analyze
    - whys_list (list): List of dictionaries containing 'why' questions and their answers
    - countermeasures (list, optional): List of countermeasures to address the root cause
    
    Returns:
    - dict: Contains the figure object and a message
    """
    if len(whys_list) < 3:
        raise ValueError("At least 3 'why' levels are required for meaningful analysis")
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('5 Whys Analysis', fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Set up the plot
    ax.set_xlim(-1, 1)
    ax.set_ylim(-len(whys_list) - 1, 1)
    ax.axis('off')
    
    # Plot the problem statement
    ax.text(0, 0, problem_statement, 
            ha='center', va='center', 
            bbox=dict(facecolor='#FFB6C1', alpha=0.5, boxstyle='round,pad=0.5'),
            fontsize=12, fontweight='bold')
    
    # Plot each "why" level
    for i, why_dict in enumerate(whys_list):
        y_pos = -(i + 1)
        
        # Plot the "why" question
        ax.text(-0.4, y_pos, why_dict['why'],
                ha='right', va='center',
                bbox=dict(facecolor='#ADD8E6', alpha=0.5, boxstyle='round,pad=0.5'),
                fontsize=10)
        
        # Plot the answer
        ax.text(0.4, y_pos, why_dict['answer'],
                ha='left', va='center',
                bbox=dict(facecolor='#90EE90', alpha=0.5, boxstyle='round,pad=0.5'),
                fontsize=10)
        
        # Draw connecting lines
        if i == 0:
            ax.plot([0, -0.4], [0, y_pos], 'k-', alpha=0.3)
            ax.plot([0, 0.4], [0, y_pos], 'k-', alpha=0.3)
        else:
            prev_y = -i
            ax.plot([-0.4, -0.4], [prev_y, y_pos], 'k-', alpha=0.3)
            ax.plot([0.4, 0.4], [prev_y, y_pos], 'k-', alpha=0.3)
    
    # Add countermeasures if provided
    if countermeasures:
        y_pos = -(len(whys_list) + 1)
        ax.text(0, y_pos, "Countermeasures:",
                ha='center', va='center',
                bbox=dict(facecolor='#FFD700', alpha=0.5, boxstyle='round,pad=0.5'),
                fontsize=10, fontweight='bold')
        
        for i, countermeasure in enumerate(countermeasures):
            y_pos = -(len(whys_list) + 2 + i)
            ax.text(0, y_pos, f"• {countermeasure}",
                    ha='center', va='center',
                    bbox=dict(facecolor='#FFD700', alpha=0.5, boxstyle='round,pad=0.5'),
                    fontsize=10)
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save the diagram
    output_path = 'output/five_whys_diagram.png'
    fig.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory
    
    # Create and save CSV data
    analysis_data = {
        'Level': list(range(1, len(whys_list) + 1)),
        'Why Question': [why_dict['why'] for why_dict in whys_list],
        'Answer': [why_dict['answer'] for why_dict in whys_list]
    }
    
    if countermeasures:
        analysis_data['Countermeasures'] = countermeasures + [''] * (len(whys_list) - len(countermeasures))
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(analysis_data)
    df.to_csv('output/five_whys_analysis.csv', index=False)
    
    return {
        'message': f"5 Whys analysis has been saved to:\n" +
                  f"- {output_path}\n" +
                  f"- output/five_whys_analysis.csv"
    }


def fishbone_diagram(categories: Dict[str, List[str]]):
    """
    Creates a fishbone (Ishikawa) diagram for cause-effect analysis.
    
    Parameters:
    - categories (dict): Dictionary mapping category names to lists of causes
    
    Returns:
    - dict: Contains status and file path information
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # Draw main spine
    ax.plot([0, 10], [0, 0], 'k-', linewidth=2)
    
    # Draw category lines
    angle = 45
    for i, (category, causes) in enumerate(categories.items()):
        x = 2 + i * 1.5
        y = 0
        dx = math.cos(math.radians(angle)) * 2
        dy = math.sin(math.radians(angle)) * 2
        
        # Draw category line
        ax.plot([x, x + dx], [y, y + dy], 'k-', linewidth=2)
        
        # Add category label
        ax.text(x + dx/2, y + dy/2, category, 
                ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        
        # Add causes
        for j, cause in enumerate(causes):
            cause_x = x + dx * 0.8
            cause_y = y + dy * 0.8 + (j - len(causes)/2) * 0.3
            ax.text(cause_x, cause_y, cause, 
                   ha='center', va='center', fontsize=8,
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    
    # Add problem statement
    ax.text(0, 0.5, "Problem", ha='right', va='center', fontweight='bold')
    
    # Remove axes
    ax.axis('off')
    
    # Save the diagram
    output_path = 'fishbone_diagram.png'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return {
        'status': 'Fishbone diagram generated successfully',
        'categories': list(categories.keys()),
        'message': f"Fishbone diagram has been saved to {output_path}"
    }


def create_ishikawa_diagram(problem_statement, categories, causes):
    """
    Creates a visual Ishikawa (Fishbone) diagram using Matplotlib.
    
    Parameters:
    - problem_statement (str): The main problem or effect being analyzed
    - categories (list): List of main categories (e.g., 'Materials', 'Methods', etc.)
    - causes (dict): Dictionary mapping categories to lists of causes
    
    Returns:
    - dict: Contains the figure object and a message
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Ishikawa (Fishbone) Diagram', fontsize=16, fontweight='bold')
    
    # Create grid layout
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Set up the plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    
    # Draw main arrow (fishbone)
    ax.arrow(-1, 0, 1.8, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Plot problem statement
    ax.text(0.8, 0.1, problem_statement, 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='#FFB6C1', alpha=0.5, boxstyle='round,pad=0.5'))
    
    # Calculate angles for category lines
    angles = np.linspace(-np.pi/2, np.pi/2, len(categories))
    
    # Plot categories and causes
    for i, (category, angle) in enumerate(zip(categories, angles)):
        # Draw category line
        x = 0.5 * np.cos(angle)
        y = 0.5 * np.sin(angle)
        ax.plot([0, x], [0, y], 'k-', alpha=0.5)
        
        # Plot category name
        ax.text(x * 1.1, y * 1.1, category,
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#ADD8E6', alpha=0.5, boxstyle='round,pad=0.3'))
        
        # Plot causes for this category
        if category in causes:
            cause_angles = np.linspace(angle - 0.2, angle + 0.2, len(causes[category]))
            for cause, cause_angle in zip(causes[category], cause_angles):
                cause_x = 0.8 * np.cos(cause_angle)
                cause_y = 0.8 * np.sin(cause_angle)
                ax.plot([x, cause_x], [y, cause_y], 'k-', alpha=0.3)
                ax.text(cause_x * 1.1, cause_y * 1.1, cause,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='#90EE90', alpha=0.5, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save the diagram
    output_path = 'output/ishikawa_diagram.png'
    fig.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory
    
    # Create and save CSV data
    # First, create a list of all causes with their categories
    analysis_data = []
    for category in categories:
        if category in causes:
            for cause in causes[category]:
                analysis_data.append({
                    'Category': category,
                    'Cause': cause
                })
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(analysis_data)
    df.to_csv('output/ishikawa_analysis.csv', index=False)
    
    return {
        'message': f"Ishikawa diagram and data have been saved to:\n" +
                  f"- {output_path}\n" +
                  f"- output/ishikawa_analysis.csv"
    }


def generate_8d_report(problem_description, team_members, containment_actions, root_cause_analysis, 
                      corrective_actions, preventive_actions, verification_results, closure_details):
    """
    Generates an 8D report for quality issues.
    
    Parameters:
    - problem_description (str): Description of the problem
    - team_members (list): List of team members involved
    - containment_actions (list): List of immediate containment actions
    - root_cause_analysis (dict): Dictionary containing analysis results and tools used
    - corrective_actions (list): List of corrective actions taken
    - preventive_actions (list): List of preventive measures
    - verification_results (list): List of verification results
    - closure_details (dict): Dictionary containing action items and lessons learned
    
    Returns:
    - dict: Contains the report data and message
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create report data
    report_data = {
        'D1 - Team Formation': {
            'Team Members': team_members,
            'Roles': ['Team Leader', 'Quality Engineer', 'Process Engineer']
        },
        'D2 - Problem Description': {
            'Description': problem_description,
            'Impact': 'Customer complaint regarding product defect'
        },
        'D3 - Containment Actions': {
            'Actions': containment_actions,
            'Timeline': 'Immediate actions taken'
        },
        'D4 - Root Cause Analysis': {
            'Results': root_cause_analysis.get('analysis_results', 'No analysis results provided'),
            'Tools Used': root_cause_analysis.get('tools_used', [])
        },
        'D5 - Corrective Actions': {
            'Actions': corrective_actions,
            'Implementation': 'Completed'
        },
        'D6 - Implement Solutions': {
            'Actions': corrective_actions,
            'Status': 'Implemented'
        },
        'D7 - Preventive Measures': {
            'Actions': preventive_actions,
            'Status': 'In Progress'
        },
        'D8 - Team Recognition': {
            'Action Items': closure_details.get('action_items', []),
            'Lessons Learned': closure_details.get('lessons_learned', [])
        }
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {'D': 'D1', 'Step': 'Team Formation', 'Details': str(report_data['D1 - Team Formation'])},
        {'D': 'D2', 'Step': 'Problem Description', 'Details': str(report_data['D2 - Problem Description'])},
        {'D': 'D3', 'Step': 'Containment Actions', 'Details': str(report_data['D3 - Containment Actions'])},
        {'D': 'D4', 'Step': 'Root Cause Analysis', 'Details': str(report_data['D4 - Root Cause Analysis'])},
        {'D': 'D5', 'Step': 'Corrective Actions', 'Details': str(report_data['D5 - Corrective Actions'])},
        {'D': 'D6', 'Step': 'Implement Solutions', 'Details': str(report_data['D6 - Implement Solutions'])},
        {'D': 'D7', 'Step': 'Preventive Measures', 'Details': str(report_data['D7 - Preventive Measures'])},
        {'D': 'D8', 'Step': 'Team Recognition', 'Details': str(report_data['D8 - Team Recognition'])}
    ])
    
    # Save to CSV
    output_path = 'output/8d_report.csv'
    df.to_csv(output_path, index=False)
    
    return {
        'status': '8D report generated successfully',
        'message': f"8D report has been saved to {output_path}",
        'data': report_data
    }

