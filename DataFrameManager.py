# --------------------------------------------
#               DataFrameManager
# --------------------------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox 
import pandas as pd 
import os 
import json 
import pickle 
from pathlib import Path

class DataFrameManager:
    """
    Class to manage loading, preprocessing, and saving DataFrames

    Needs import subprocess in script

    To open GUI: 
    - subprocess.Popen(["Python", "DataFrameManager.py"])
    """
    
    def __init__(self, config_path="df_manager_config.json", data_path="saved_dataframes"):
        self.config_path = config_path
        self.data_path = data_path
        self.dataframes = {}
        self.file_groups = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
        
        # Load previous configuration if exists
        self.load_config()
    
    def load_config(self):
        """Load saved configuration if exists"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.file_groups = config.get('file_groups', {})
            except Exception as e:
                print(f"Error loading config: {e}")
                self.file_groups = {}
    
    def save_config(self):
        """Save current configuration"""
        config = {
            'file_groups': self.file_groups
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def load_dataframe_group(self, group_name):
        """Load a group of files into a combined DataFrame"""
        if group_name not in self.file_groups or not self.file_groups[group_name]:
            return None
        
        file_paths = self.file_groups[group_name]
        
        # Check if all files still exist
        missing_files = [f for f in file_paths if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
        
        # Load and combine DataFrames
        dfs = []
        for file_path in file_paths:
            file_name = os.path.basename(file_path).split(".")[0] #extract filename (without .rowOut)
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            elif file_path.lower().endswith('.rowout'):
                with open(file_path, "r") as f:
                  headers = f.readline().strip().split()
                df=pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=headers, index_col=False)
                df["FILE_NAME"] = file_name #add file name column 
                #Move filename to first column
                if "FILE_NAME" in df.columns:
                    cols = ["FILE_NAME"] + [col for col in df.columns if col != "FILE_NAME"]
                    df = df[cols] # reorder columns

            else:
                continue
            dfs.append(df)
        
        if not dfs:
            return None
        
        # Combine DataFrames (old way)
        # combined_df = pd.concat(dfs, ignore_index=True)
        # return combined_df
    
        #bug fix
        try:
            # combine dataFrame
            combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df
        except ValueError as e:
            if "duplicate names" in str(e).lower():
                # fixing duplicate names bug
                renamed_dfs = []
                for i, df in enumerate(dfs):
                    df_copy = df.copy()
                    df_copy.columns = [f"{col}_{i}" for col in df.columns]
                    renamed_dfs.append(df_copy)

                combined_df = pd.concat(renamed_dfs, ignore_index=True)
                return combined_df
            else:
                raise
    
    def process_dataframe(self, df, handle_nulls='drop'):
        """Process DataFrame by handling null values"""
        if df is None:
            return None
        
        if handle_nulls == 'drop':
            return df.dropna()
        elif handle_nulls == 'zero':
            return df.fillna(0)
        else:
            return df
    
    def save_dataframe(self, df, name):
        """Save DataFrame to disk"""
        if df is None:
            return False
        
        file_path = os.path.join(self.data_path, f"{name}.pkl")
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(df, f)
            return True
        except Exception as e:
            print(f"Error saving DataFrame: {e}")
            return False
    
    def load_saved_dataframe(self, name):
        """Load a saved DataFrame from disk"""
        file_path = os.path.join(self.data_path, f"{name}.pkl")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            return df
        except Exception as e:
            print(f"Error loading DataFrame: {e}")
            return None


class FileManagerApp:
    """Main application window for file selection and DataFrame management"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DataFrame Manager")
        self.root.geometry("800x600")
        
        self.df_manager = DataFrameManager()
        
        self.create_widgets()
        self.update_group_listbox()
        self.update_saved_df_listbox()
    
    def create_widgets(self):
        """Create all UI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame - File Groups
        left_frame = ttk.LabelFrame(main_frame, text="File Groups", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Group management
        group_frame = ttk.Frame(left_frame)
        group_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(group_frame, text="Group Name:").pack(side=tk.LEFT)
        self.group_name_var = tk.StringVar()
        ttk.Entry(group_frame, textvariable=self.group_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(group_frame, text="New Group", command=self.create_group).pack(side=tk.LEFT)
        
        # Group selection
        group_select_frame = ttk.Frame(left_frame)
        group_select_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(group_select_frame, text="Select Group:").pack(anchor=tk.W)
        self.group_listbox = tk.Listbox(group_select_frame)
        self.group_listbox.pack(fill=tk.BOTH, expand=True)
        self.group_listbox.bind('<<ListboxSelect>>', self.on_group_select)
        
        group_btn_frame = ttk.Frame(left_frame)
        group_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(group_btn_frame, text="Add Files", command=self.add_files_to_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(group_btn_frame, text="Remove Files", command=self.remove_files_from_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(group_btn_frame, text="Delete Group", command=self.delete_group).pack(side=tk.LEFT, padx=2)
        
        # Files in group
        files_frame = ttk.Frame(left_frame)
        files_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(files_frame, text="Files in Group:").pack(anchor=tk.W)
        self.files_listbox = tk.Listbox(files_frame)
        self.files_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Right frame - DataFrame Operations
        right_frame = ttk.LabelFrame(main_frame, text="DataFrame Operations", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # DataFrame operation controls
        ops_frame = ttk.Frame(right_frame)
        ops_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ops_frame, text="Handle Null Values:").pack(side=tk.LEFT)
        self.null_handling = tk.StringVar(value="drop")
        null_dropdown = ttk.Combobox(ops_frame, textvariable=self.null_handling, state="readonly",
                                    values=["drop", "zero", "keep"])
        null_dropdown.pack(side=tk.LEFT, padx=5)
        
        df_name_frame = ttk.Frame(right_frame)
        df_name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(df_name_frame, text="DataFrame Name:").pack(side=tk.LEFT)
        self.df_name_var = tk.StringVar()
        ttk.Entry(df_name_frame, textvariable=self.df_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        df_ops_btn_frame = ttk.Frame(right_frame)
        df_ops_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(df_ops_btn_frame, text="Process & Save", command=self.process_and_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(df_ops_btn_frame, text="Preview Data", command=self.preview_data).pack(side=tk.LEFT, padx=2)
        
        # Saved DataFrames
        saved_df_frame = ttk.Frame(right_frame)
        saved_df_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(saved_df_frame, text="Saved DataFrames:").pack(anchor=tk.W)
        self.saved_df_listbox = tk.Listbox(saved_df_frame)
        self.saved_df_listbox.pack(fill=tk.BOTH, expand=True)
        
        saved_df_btn_frame = ttk.Frame(right_frame)
        saved_df_btn_frame.pack(fill=tk.X)
        
        ttk.Button(saved_df_btn_frame, text="Load DataFrame", command=self.load_dataframe).pack(side=tk.LEFT, padx=2)
        ttk.Button(saved_df_btn_frame, text="Delete DataFrame", command=self.delete_dataframe).pack(side=tk.LEFT, padx=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_group_listbox(self):
        """Update the group listbox with current groups"""
        self.group_listbox.delete(0, tk.END)
        for group_name in self.df_manager.file_groups:
            self.group_listbox.insert(tk.END, group_name)
    
    def update_files_listbox(self, group_name=None):
        """Update the files listbox with files in the selected group"""
        self.files_listbox.delete(0, tk.END)
        
        if not group_name:
            selected = self.group_listbox.curselection()
            if not selected:
                return
            group_name = self.group_listbox.get(selected[0])
        
        if group_name in self.df_manager.file_groups:
            file_paths = self.df_manager.file_groups[group_name]
            for file_path in file_paths:
                # Just show the filename, not the full path
                self.files_listbox.insert(tk.END, os.path.basename(file_path))
    
    def update_saved_df_listbox(self):
        """Update the saved DataFrame listbox"""
        self.saved_df_listbox.delete(0, tk.END)
        
        data_dir = Path(self.df_manager.data_path)
        if data_dir.exists():
            for file_path in data_dir.glob("*.pkl"):
                self.saved_df_listbox.insert(tk.END, file_path.stem)
    
    def on_group_select(self, event):
        """Handler for group selection"""
        self.update_files_listbox()
    
    def create_group(self):
        """Create a new file group"""
        group_name = self.group_name_var.get().strip()
        
        if not group_name:
            messagebox.showerror("Error", "Group name cannot be empty")
            return
        
        if group_name in self.df_manager.file_groups:
            messagebox.showerror("Error", f"Group '{group_name}' already exists")
            return
        
        self.df_manager.file_groups[group_name] = []
        self.df_manager.save_config()
        self.update_group_listbox()
        self.status_var.set(f"Created group: {group_name}")
    
    def add_files_to_group(self):
        """Add files to the selected group"""
        selected = self.group_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a group first")
            return
        
        group_name = self.group_listbox.get(selected[0])
        
        file_paths = filedialog.askopenfilenames(
            title="Select data files",
            filetypes=[("Data files", "*.csv;*.xls;*.xlsx"), ("All files", "*.*")]
        )
        
        if not file_paths:
            return
        
        # Add new files to the group
        for path in file_paths:
            if path not in self.df_manager.file_groups[group_name]:
                self.df_manager.file_groups[group_name].append(path)
        
        self.df_manager.save_config()
        self.update_files_listbox(group_name)
        self.status_var.set(f"Added {len(file_paths)} files to {group_name}")
    
    def remove_files_from_group(self):
        """Remove selected files from the group"""
        group_selected = self.group_listbox.curselection()

        if not group_selected:
            messagebox.showerror("Error", "Please select a group first")
            return

        group_name = self.group_listbox.get(group_selected[0])

        # Creates popup window for file selection
        popup = tk.Toplevel(self.root)
        popup.title("Select Files to Remove")
        popup.geometry("400x300")
        popup.grab_set()  # Make it modal

        # Instructions
        ttk.Label(popup, text=f"Select files to remove from group '{group_name}':", padding=10).pack(anchor=tk.W)

        # listbox with multiple selection enabled
        file_listbox = tk.Listbox(popup, selectmode=tk.MULTIPLE)
        file_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Populate the listbox with filenames
        for path in self.df_manager.file_groups[group_name]:
            file_listbox.insert(tk.END, os.path.basename(path))

        # Button frame
        button_frame = ttk.Frame(popup)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        # Function to handle removing files
        def do_remove():
            selected_indices = file_listbox.curselection()
            if not selected_indices:
                messagebox.showinfo("Info", "No files selected")
                return

            selected_filenames = [file_listbox.get(i) for i in selected_indices]

            # Find and remove the actual file paths
            paths_to_remove = []
            for filename in selected_filenames:
                for path in self.df_manager.file_groups[group_name]:
                    if os.path.basename(path) == filename:
                        paths_to_remove.append(path)
                        break
                    
            # Remove the paths
            for path in paths_to_remove:
                self.df_manager.file_groups[group_name].remove(path)

            self.df_manager.save_config()
            self.update_files_listbox(group_name)
            self.status_var.set(f"Removed {len(paths_to_remove)} files from '{group_name}'")
            popup.destroy()

        # Buttons
        ttk.Button(button_frame, text="Remove Selected", command=do_remove).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=popup.destroy).pack(side=tk.RIGHT, padx=5)

    def delete_group(self):
        """Delete the selected group"""
        selected = self.group_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a group first")
            return
        
        group_name = self.group_listbox.get(selected[0])
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete group '{group_name}'?"):
            del self.df_manager.file_groups[group_name]
            self.df_manager.save_config()
            self.update_group_listbox()
            self.files_listbox.delete(0, tk.END)
            self.status_var.set(f"Deleted group: {group_name}")
    
    def process_and_save(self):
        """Process and save the selected group as a DataFrame"""
        selected = self.group_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a group first")
            return
        
        group_name = self.group_listbox.get(selected[0])
        df_name = self.df_name_var.get().strip()
        
        if not df_name:
            messagebox.showerror("Error", "DataFrame name cannot be empty")
            return
        
        try:
            # Load and process DataFrame
            df = self.df_manager.load_dataframe_group(group_name)
            if df is None or df.empty:
                messagebox.showerror("Error", "No valid data files in this group")
                return
            
            # Handle null values
            handling = self.null_handling.get()
            df = self.df_manager.process_dataframe(df, handle_nulls=handling)
            
            # Save DataFrame
            success = self.df_manager.save_dataframe(df, df_name)
            if success:
                self.update_saved_df_listbox()
                self.status_var.set(f"Processed and saved DataFrame '{df_name}' ({len(df)} rows)")
            else:
                messagebox.showerror("Error", "Failed to save DataFrame")
        
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def preview_data(self):
        """Preview data from the selected group"""
        selected = self.group_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a group first")
            return
        
        group_name = self.group_listbox.get(selected[0])
        
        try:
            # Load DataFrame
            df = self.df_manager.load_dataframe_group(group_name)
            if df is None or df.empty:
                messagebox.showerror("Error", "No valid data files in this group")
                return
            
            # Handle null values
            handling = self.null_handling.get()
            df = self.df_manager.process_dataframe(df, handle_nulls=handling)
            
            # Create preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title(f"Preview: {group_name}")
            preview_window.geometry("800x600")
            
            # DataFrame info
            info_frame = ttk.Frame(preview_window, padding="10")
            info_frame.pack(fill=tk.X)
            
            info_text = (
                f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                f"Columns: {', '.join(df.columns)}\n"
                f"Null handling: {handling}"
            )
            ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)
            
            # DataFrame preview (first 50 rows)
            preview_frame = ttk.Frame(preview_window, padding="10")
            preview_frame.pack(fill=tk.BOTH, expand=True)
            
            preview_text = tk.Text(preview_frame, wrap=tk.NONE)
            preview_text.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars
            y_scrollbar = ttk.Scrollbar(preview_text, orient=tk.VERTICAL, command=preview_text.yview)
            preview_text.configure(yscrollcommand=y_scrollbar.set)
            y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            x_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=preview_text.xview)
            preview_text.configure(xscrollcommand=x_scrollbar.set)
            x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Display DataFrame head
            preview_content = df.head(50).to_string(index=True)
            preview_text.insert(tk.END, preview_content)
            preview_text.configure(state='disabled')  # Make it read-only
            
            self.status_var.set(f"Previewing data from group: {group_name}")
        
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_dataframe(self):
        """Load a saved DataFrame into a Python variable and provide code to access it"""
        selected = self.saved_df_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a saved DataFrame first")
            return
        
        df_name = self.saved_df_listbox.get(selected[0])
        
        # Create code window
        code_window = tk.Toplevel(self.root)
        code_window.title(f"Load DataFrame: {df_name}")
        code_window.geometry("600x300")
        
        # Code frame
        code_frame = ttk.Frame(code_window, padding="10")
        code_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(code_frame, text="Use this code to load your DataFrame:").pack(anchor=tk.W, pady=(0, 10))
        
        code_text = tk.Text(code_frame, height=10)
        code_text.pack(fill=tk.BOTH, expand=True)
        
        # Generate code insert for copy button
        code = f"""import pickle

# Load the DataFrame
with open(r"{os.path.abspath(os.path.join(self.df_manager.data_path, df_name + '.pkl'))}", 'rb') as f:
    {df_name}_df = pickle.load(f)

# Now dataFrame {df_name}_df is ready to use.
print({df_name}_df.head())
"""
        code_text.insert(tk.END, code)
        
        # Copy button
        def copy_to_clipboard():
            code_window.clipboard_clear()
            code_window.clipboard_append(code_text.get("1.0", tk.END))
            self.status_var.set("Code copied to clipboard")
        
        ttk.Button(code_frame, text="Copy to Clipboard", command=copy_to_clipboard).pack(pady=10)
        
        self.status_var.set(f"Prepared code to load DataFrame: {df_name}")
    
    def delete_dataframe(self):
        """Delete a saved DataFrame"""
        selected = self.saved_df_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select a saved DataFrame first")
            return
        
        df_name = self.saved_df_listbox.get(selected[0])
        file_path = os.path.join(self.df_manager.data_path, f"{df_name}.pkl")
        
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete DataFrame '{df_name}'?"):
            try:
                os.remove(file_path)
                self.update_saved_df_listbox()
                self.status_var.set(f"Deleted DataFrame: {df_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete DataFrame: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FileManagerApp(root)
    root.mainloop()
