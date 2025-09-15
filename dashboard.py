# --------------------------------------------
#               Dashboard 
# --------------------------------------------
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output 
import re

class Dashboard:
    """
    Dashboard for data visualization.
    
    Supported Visualizations (each in its own tab):
      1. 2D Scatter Plot (with selectable X, Y, and color variables) -- updates in place.
      2. 3D Scatter Plot (with selectable X, Y, Z, and color variables) -- updates in place.
      3. 3D Surface Plot (from a pivot of the data) -- updates in place.
      4. Parallel Coordinates Plot (multidimensional visualization) -- updates in place.
      5. Correlation Heatmap (based on selected numeric variables) -- updates in place.
      6. Data Table (interactive with filtering and sorting)
      7. Statistics Panel (summary statistics for selected variables)
    """
    
    def __init__(self, df):
        self.df = df
        self.current_theme = "plotly"
        
        # Get numeric columns for plots that need numbers
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create tooltip dictionary for variables (detect units from column names)
        self.tooltips = {}
        for col in df.columns:
            # Extract units if in parentheses or brackets
            unit_match = re.search(r'[\(\[]([^\)\]]+)[\)\]]', col)
            if unit_match:
                unit = unit_match.group(1)
                base_name = col.split('(')[0].strip() if '(' in col else col.split('[')[0].strip()
                self.tooltips[col] = f"{base_name} [{unit}]"
            else:
                self.tooltips[col] = col
        
        # --- Create theme selector ---
        self.theme_selector = widgets.Dropdown(
            options=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
            value="plotly",
            description="Theme:",
            layout=widgets.Layout(width='200px') # Layout accepts CSS style syntax (e.g. height='50px', border='solid 2px blue', margin='10px')
        )
        self.theme_selector.observe(self.update_theme, names='value')
        
        # --- Create persistent FigureWidgets for other plots ---
        default_x = self.numeric_columns[0]
        default_y = self.numeric_columns[1] if len(self.numeric_columns) > 1 else self.numeric_columns[0]
        default_z = self.numeric_columns[2] if len(self.numeric_columns) > 2 else self.numeric_columns[0]
        default_color = self.numeric_columns[2] if len(self.numeric_columns) > 2 else self.numeric_columns[0]
        
        self.scatter_fig = go.FigureWidget(self.create_scatter_figure(default_x, default_y, default_color))
        self.scatter3d_fig = go.FigureWidget(self.create_scatter3d_figure(default_x, default_y, default_z, default_color))
        self.surface_fig = go.FigureWidget(self.create_surface_figure(default_x, default_y, default_z))
        self.parallel_fig = go.FigureWidget(self.create_parallel_figure(list(self.numeric_columns[:5]), title="Parallel Coordinates Plot"))
        self.heatmap_fig = go.FigureWidget(self.create_heatmap_figure(list(self.numeric_columns[:5]), title="Correlation Heatmap"))
        
        # Define a common container layout for each tab
        container_layout = widgets.Layout(width='100%', height='750px')
        # Horizontal scroll (scroll hidden) for data table
        scroll_layout = widgets.Layout(overflow='scroll', width='100%', height='75%', flex_flow='row', display='flex')
        
        # --- 2D Scatter Plot Tab ---
        self.scatter_x_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_x, description="X Axis:"
        )
        self.scatter_y_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_y, description="Y Axis:"
        )
        self.scatter_color_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_color, description="Color:"
        )
        self.scatter_size_dropdown = widgets.Dropdown(
            options=["None"] + self.numeric_columns, value="None", description="Size:"
        )
        
        # Add regression line option
        self.scatter_trend = widgets.Checkbox(
            value=False,
            description='Show Trendline',
            indent=False
        )
        
        self.scatter_help = widgets.HTML(
            value="<p><i>Compare relationships between two variables.</i></p>"
        )
        
        for w in [self.scatter_x_dropdown, self.scatter_y_dropdown, self.scatter_color_dropdown, 
                 self.scatter_size_dropdown, self.scatter_trend]:
            w.observe(self.update_scatter, names='value')
            
        self.scatter_controls = widgets.VBox([
            self.scatter_help,
            widgets.HBox([self.scatter_x_dropdown, self.scatter_y_dropdown]),
            widgets.HBox([self.scatter_color_dropdown, self.scatter_size_dropdown, self.scatter_trend])
        ])
        
        self.scatter_widget = widgets.VBox(
            [self.scatter_controls, self.scatter_fig],
            layout=container_layout
        )
        
        # --- 3D Scatter Plot Tab ---
        self.scatter3d_x_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_x, description="X Axis:"
        )
        self.scatter3d_y_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_y, description="Y Axis:"
        )
        self.scatter3d_z_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_z, description="Z Axis:"
        )
        self.scatter3d_color_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_color, description="Color:"
        )
        
        # Add size option for 3D scatter
        self.scatter3d_size_dropdown = widgets.Dropdown(
            options=["None"] + self.numeric_columns, value="None", description="Size:"
        )
        # display(self.scatter3d_help) to show
        self.scatter3d_help = widgets.HTML(
            value="<p><i>Visualize relationships between three variables.</i></p>"
        )
        
        for w in [self.scatter3d_x_dropdown, self.scatter3d_y_dropdown, self.scatter3d_z_dropdown, 
                 self.scatter3d_color_dropdown, self.scatter3d_size_dropdown]:
            w.observe(self.update_scatter3d, names='value')
            
        self.scatter3d_controls = widgets.VBox([
            self.scatter3d_help,
            widgets.HBox([self.scatter3d_x_dropdown, self.scatter3d_y_dropdown, self.scatter3d_z_dropdown]),
            widgets.HBox([self.scatter3d_color_dropdown, self.scatter3d_size_dropdown])
        ])
        
        self.scatter3d_widget = widgets.VBox(
            [self.scatter3d_controls, self.scatter3d_fig],
            layout=container_layout
        )
        
        # --- 3D Surface Plot Tab ---
        self.surface_x_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_x, description="X Axis:"
        )
        self.surface_y_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_y, description="Y Axis:"
        )
        self.surface_z_dropdown = widgets.Dropdown(
            options=self.numeric_columns, value=default_z, description="Z Axis:"
        )
        
        # Add surface color options
        self.surface_colorscale = widgets.Dropdown(
            options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Jet", "Rainbow"],
            value="Viridis",
            description="Color Scale:"
        )
        
        self.surface_help = widgets.HTML(
            value="<p><i>Visualize 3D surfaces created from pivoting X-Y grid with Z values.</i></p>"
        )
        
        for w in [self.surface_x_dropdown, self.surface_y_dropdown, self.surface_z_dropdown, self.surface_colorscale]:
            w.observe(self.update_surface, names='value')
            
        self.surface_controls = widgets.VBox([
            self.surface_help,
            widgets.HBox([self.surface_x_dropdown, self.surface_y_dropdown, self.surface_z_dropdown]),
            widgets.HBox([self.surface_colorscale])
        ])
        
        self.surface_widget = widgets.VBox(
            [self.surface_controls, self.surface_fig],
            layout=container_layout
        )
        
        # --- Parallel Coordinates Plot Tab ---
        self.parallel_vars = widgets.SelectMultiple(
            options=self.numeric_columns,
            value=tuple(self.numeric_columns[:5] if len(self.numeric_columns) >= 5 else self.numeric_columns),
            description="Variables:",
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        self.parallel_color = widgets.Dropdown(
            options=["None"] + self.numeric_columns,
            value=self.numeric_columns[0] if self.numeric_columns else "None",
            description="Color by:"
        )
        
        self.parallel_help = widgets.HTML(
            value="<p><i>Visualize multiple variables at once, with each vertical axis representing one variable.</i></p>"
        )
        
        self.parallel_button = widgets.Button(description="Update Parallel Plot")
        self.parallel_button.on_click(self.update_parallel)
        
        self.parallel_controls = widgets.VBox([
            self.parallel_help,
            widgets.HBox([self.parallel_vars, widgets.VBox([self.parallel_color, self.parallel_button])])
        ])
        
        self.parallel_widget = widgets.VBox(
            [self.parallel_controls, self.parallel_fig],
            layout=container_layout
        )
        
        # --- Correlation Heatmap Tab ---
        self.heatmap_vars = widgets.SelectMultiple(
            options=self.numeric_columns,
            value=tuple(self.numeric_columns[:5] if len(self.numeric_columns) >= 5 else self.numeric_columns),
            description="Variables:",
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        self.heatmap_method = widgets.Dropdown(
            options=["pearson", "spearman", "kendall"],
            value="pearson",
            description="Correlation:"
        )
        
        self.heatmap_colorscale = widgets.Dropdown(
            options=["RdBu_r", "Viridis", "Plasma", "Blues", "Reds"],
            value="RdBu_r",
            description="Colors:"
        )
        
        self.heatmap_help = widgets.HTML(
            value="<p><i>Visualize correlation coefficients between variables. Blue = positive correlation, Red = negative correlation.</i></p>"
        )
        
        self.heatmap_button = widgets.Button(description="Update Heatmap")
        self.heatmap_button.on_click(self.update_heatmap)
        
        self.heatmap_controls = widgets.VBox([
            self.heatmap_help,
            widgets.HBox([
                self.heatmap_vars, 
                widgets.VBox([
                    self.heatmap_method,
                    self.heatmap_colorscale,
                    self.heatmap_button
                ])
            ])
        ])
        
        self.heatmap_widget = widgets.VBox(
            [self.heatmap_controls, self.heatmap_fig],
            layout=container_layout
        )
        
        # --- Data Table Tab ---
        self.table_output = widgets.Output()
        self.table_rows = widgets.BoundedIntText(
            value=10,
            min=5,
            max=100,
            step=5,
            description='Rows:',
            layout=widgets.Layout(width='150px')
        )
        
        self.table_filter_col = widgets.Dropdown(
            options=["None"] + list(df.columns),
            value="None",
            description="Filter by:",
            layout=widgets.Layout(width='200px')
        )
        
        self.table_filter_text = widgets.Text(
            value='',
            placeholder='Filter value...',
            description='',
            disabled=True,
            layout=widgets.Layout(width='200px')
        )
        
        # Enable/disable filter text based on selection
        def update_filter_status(change):
            self.table_filter_text.disabled = change.new == "None"
        
        self.table_filter_col.observe(update_filter_status, names='value')
        
        self.table_button = widgets.Button(description="Update Table")
        self.table_button.on_click(self.update_table)
        
        self.table_help = widgets.HTML(
            value="<p><i>Scrollable data table with filtering capabilities.</i>"
            "<br><i>Min & Max Highlighted</i></p>"
        )
        
        self.table_controls = widgets.VBox([
            self.table_help,
            widgets.HBox([
                self.table_rows,
                self.table_filter_col,
                self.table_filter_text,
                self.table_button
            ])
        ])
        self.scroll = widgets.Box(
            [self.table_output],
            layout=scroll_layout
        )

        self.table_widget = widgets.VBox(
            [self.table_controls, self.scroll],
            layout=container_layout
        )
        
        # --- Statistics Panel Tab ---
        self.stats_output = widgets.Output()
        self.stats_vars = widgets.SelectMultiple(
            options=self.numeric_columns,
            value=tuple(self.numeric_columns[:3] if len(self.numeric_columns) >= 3 else self.numeric_columns),
            description="Variables:",
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        self.stats_categorical = widgets.Dropdown(
            options=["None"] + self.categorical_columns,
            value="None",
            description="Group by:"
        )
        
        self.stats_button = widgets.Button(description="Calculate Statistics")
        self.stats_button.on_click(self.update_statistics)
        
        self.stats_help = widgets.HTML(
            value="<p><i>Generate summary statistics for selected variables.</i><br><i>🐛 Still needs bugs fixed 🐛</i></p>"
        )
        
        self.stats_controls = widgets.VBox([
            self.stats_help,
            widgets.HBox([
                self.stats_vars,
                widgets.VBox([
                    self.stats_categorical,
                    self.stats_button
                ])
            ])
        ])
        
        self.stats_widget = widgets.VBox(
            [self.stats_controls, self.stats_output],
            layout=container_layout
        )
        
        # --- Assemble all tabs with theme selector at top ---
        self.header = widgets.HBox([
            widgets.HTML("<h2>DataFrame Dashboard</h2>"),
            self.theme_selector
        ])
        
        self.tab = widgets.Tab(children=[
            self.scatter_widget,
            self.scatter3d_widget,
            self.surface_widget,
            self.parallel_widget,
            self.heatmap_widget,
            self.table_widget,
            self.stats_widget
        ])
        
        tab_titles = [
            "2D Scatter Plot", 
            "3D Scatter Plot", 
            "3D Surface Plot", 
            "Parallel Coordinates", 
            "Correlation Heatmap",
            "Data Table",
            "Statistics"
        ]
        
        for i, t in enumerate(tab_titles):
            self.tab.set_title(i, t)
        
        # Main layout
        self.main_layout = widgets.VBox([self.header, self.tab])
        
        # Initial update for the data table (the others are already created)
        self.update_table(None) # dataTable
        # self.update_statistics(None) # was annoying
    
    # Helper function to update a FigureWidget in place
    def update_figurewidget(self, fig_widget, new_fig):
        """Helper to update a FigureWidget in place"""
        with fig_widget.batch_update():
            # Remove all existing traces
            fig_widget.data = []
            # Add new traces from the new figure
            for trace in new_fig.data:
                fig_widget.add_trace(trace)
            # Update layout
            fig_widget.layout = new_fig.layout
            
    def update_theme(self, change):
        """Update the theme for all plots"""
        self.current_theme = change.new
        
        # Update all figure widgets
        self.update_scatter(None)
        self.update_scatter3d(None)
        self.update_surface(None)
        self.update_parallel(None)
        self.update_heatmap(None)
            
    def create_scatter3d_figure(self, x_col, y_col, z_col, color_col, size_col="None"):
        """Create a 3D scatter plot with proper size parameter handling"""

        # Create base keyword arguments for the plot
        plot_kwargs = {
            'x': x_col,
            'y': y_col,
            'z': z_col,
            'color': color_col,
            'title': "3D Scatter Plot",
            'labels': {
                x_col: self.tooltips.get(x_col, x_col),
                y_col: self.tooltips.get(y_col, y_col),
                z_col: self.tooltips.get(z_col, z_col),
                color_col: self.tooltips.get(color_col, color_col)
            },
            'template': self.current_theme,
        }

        # Add size parameter if provided and not "None"
        if size_col != "None":
            # Ensure size values are positive
            temp_df = self.df.copy()
            temp_df[f'{size_col}_abs'] = temp_df[size_col].abs()

            # Add size parameter and update dataset
            plot_kwargs['size'] = f'{size_col}_abs'
            plot_kwargs['size_max'] = 20  # Set maximum size
            plot_kwargs['labels'][f'{size_col}_abs'] = f"|{self.tooltips.get(size_col, size_col)}|"
            plot_kwargs['hover_data'] = {f'{size_col}_abs': False, size_col: True}  # Show original in hover

            # Create figure using the modified dataset
            fig = px.scatter_3d(temp_df, **plot_kwargs)

            # Debug print
            # print(f"Using size column: {size_col}, with values range: {temp_df[size_col].min()} to {temp_df[size_col].max()}")
            # print(f"Absolute size values range: {temp_df[f'{size_col}_abs'].min()} to {temp_df[f'{size_col}_abs'].max()}")
        else:
            # Create figure without size parameter
            fig = px.scatter_3d(self.df, **plot_kwargs)

        # Update layout
        fig.update_layout(
            width=800, 
            height=700,
            autosize=True, 
            margin=dict(l=50, r=50, t=50, b=50),
            scene=dict(
                aspectmode='cube',
                xaxis_title=self.tooltips.get(x_col, x_col),
                yaxis_title=self.tooltips.get(y_col, y_col),
                zaxis_title=self.tooltips.get(z_col, z_col)
            )
        )

        return fig


    def create_scatter_figure(self, x_col, y_col, color_col, size_col="None", show_trend=False):
        """Create a 2D scatter plot with optional trendline"""
        if size_col == "None":
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, 
                           title="2D Scatter Plot",
                           labels={
                               x_col: self.tooltips.get(x_col, x_col),
                               y_col: self.tooltips.get(y_col, y_col),
                               color_col: self.tooltips.get(color_col, color_col)
                           },
                           template=self.current_theme)
        else:
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, size=size_col,
                           title="2D Scatter Plot",
                           labels={
                               x_col: self.tooltips.get(x_col, x_col),
                               y_col: self.tooltips.get(y_col, y_col),
                               color_col: self.tooltips.get(color_col, color_col),
                               size_col: self.tooltips.get(size_col, size_col)
                           },
                           template=self.current_theme)
        
        # Add trendline if clicked
        if show_trend:
            try:
                # Calculate regression line
                x_values = self.df[x_col].values
                y_values = self.df[y_col].values
                
                # Filter out any NaN values
                mask = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[mask]
                y_values = y_values[mask]
                
                if len(x_values) > 1:  # Need at least 2 points for a line
                    z = np.polyfit(x_values, y_values, 1)
                    p = np.poly1d(z)
                    
                    # Create x values spanning the range
                    x_line = np.linspace(min(x_values), max(x_values), 100)
                    y_line = p(x_line)
                    
                    # Add the trendline
                    fig.add_trace(go.Scatter(
                        x=x_line, 
                        y=y_line, 
                        mode='lines', 
                        name=f'y = {z[0]:.3f}x + {z[1]:.3f}',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Add R-squared value
                    residuals = y_values - p(x_values)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_values - np.mean(y_values))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    fig.add_annotation(
                        x=0.05, 
                        y=0.95, 
                        xref="paper", 
                        yref="paper",
                        text=f"R² = {r_squared:.3f}",
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="black",
                        borderwidth=1
                    )
            except Exception as e:
                print(f"Error adding trendline: {str(e)}")
        
        fig.update_layout(autosize=True, margin=dict(l=50, r=50, t=50, b=50))
        return fig


    def create_surface_figure(self, x_col, y_col, z_col, colorscale="Viridis"):
         """Create a 3D surface plot with error handling and colorscale options"""
         try:
             # Create more advanced pivot with binning for continuous variables if needed
             if len(self.df[x_col].unique()) > 20 or len(self.df[y_col].unique()) > 20:
                 # Use binning for variables with too many unique values
                 x_bins = min(20, len(self.df[x_col].unique()))
                 y_bins = min(20, len(self.df[y_col].unique()))

                 # Create binned columns
                 self.df[f'{x_col}_binned'] = pd.qcut(self.df[x_col], q=x_bins, duplicates='drop')
                 self.df[f'{y_col}_binned'] = pd.qcut(self.df[y_col], q=y_bins, duplicates='drop')

                 # Use bin midpoints for pivot
                 x_midpoints = self.df.groupby(f'{x_col}_binned')[x_col].mean()
                 y_midpoints = self.df.groupby(f'{y_col}_binned')[y_col].mean()

                 # Create mapping dictionaries
                 x_map = {b: m for b, m in zip(x_midpoints.index, x_midpoints.values)}
                 y_map = {b: m for b, m in zip(y_midpoints.index, y_midpoints.values)}

                 # Apply mapping to get numeric values
                 self.df[f'{x_col}_mid'] = self.df[f'{x_col}_binned'].map(x_map)
                 self.df[f'{y_col}_mid'] = self.df[f'{y_col}_binned'].map(y_map)

                 # Create pivot table with binned data
                 pivot_table = self.df.pivot_table(
                     index=f'{y_col}_mid', 
                     columns=f'{x_col}_mid', 
                     values=z_col, 
                     aggfunc='mean'
                 )

                 # Clean up temporary columns
                 self.df = self.df.drop([f'{x_col}_binned', f'{y_col}_binned', 
                                        f'{x_col}_mid', f'{y_col}_mid'], axis=1)
             else:
                 # Create regular pivot if variables have reasonable number of unique values
                 pivot_table = self.df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')

             # Replace any non-finite numbers with 0
             z_values = pivot_table.values
             z_values = np.where(np.isfinite(z_values), z_values, 0)

             # Round decimals to 4 places to avoid JSON float issues
             z_values = np.around(z_values, decimals=4)

             # Improved hover template to display variable names
             hovertemplate = f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z}}<extra></extra>"

             # Create surface with selected colorscale
             fig = go.Figure(data=[go.Surface(
                 z=z_values,
                 x=pivot_table.columns.values,
                 y=pivot_table.index.values,
                 colorscale=colorscale.lower(),
                 hovertemplate=hovertemplate
             )])

             # Add contour projections for better visibility
             fig.update_traces(
                 contours_z=dict(
                     show=True,
                     usecolormap=True,
                     highlightcolor="white",
                     project_z=True
                 )
             )

             # Improved layout with axis labels from tooltips
             fig.update_layout(
                 width=800,
                 height=700,
                 title="3D Surface Plot",
                 autosize=True,
                 margin=dict(l=50, r=50, t=50, b=50),
                 scene=dict(
                     xaxis=dict(title=self.tooltips.get(x_col, x_col)),
                     yaxis=dict(title=self.tooltips.get(y_col, y_col)),
                     zaxis=dict(title=self.tooltips.get(z_col, z_col)),
                     aspectmode='cube'
                 ),
                 template=self.current_theme
             )
             # Download png button for plots
             # plotly download button is working now so this feature is no longer necessary

             # Add download button
            #  fig.update_layout(
            #      updatemenus=[
            #          dict(
            #              type="buttons",
            #              direction="left",
            #              buttons=[
            #                  dict(
            #                      args=["toImage", {"format": "png", "width": 800, "height": 700}],
            #                      label="Download PNG",
            #                      method="relayout"
            #                  )
            #              ],
            #              pad={"r": 10, "t": 10},
            #              showactive=False,
            #              x=0.31,
            #              xanchor="left",
            #              y=1.1,
            #              yanchor="top"
            #          )
            #      ]
            #  )

         except Exception as e:
             # error message
             fig = go.Figure()
             fig.add_annotation(
                 text=f"Error creating surface plot: {str(e)}\nTry selecting different variables or reduce resolution.",
                 showarrow=False,
                 font=dict(size=14, color="red")
             )

         return fig

    def create_parallel_figure(self, dimensions, title="Parallel Coordinates Plot", color_var="None"):
        """Create an enhanced parallel coordinates plot with proper color handling"""
        if not dimensions:
            # Handle empty dimensions case
            fig = go.Figure()
            fig.add_annotation(
                text="Please select at least one dimension for the parallel coordinates plot.",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

        try:
            # Create base figure without color mapping first
            if color_var == "None" or color_var not in self.df.columns:
                # Create standard parallel coordinates without color mapping
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(color='blue', colorscale='Blues', 
                                 showscale=False),
                        dimensions=[
                            dict(
                                range=[self.df[dim].min(), self.df[dim].max()],
                                label=self.tooltips.get(dim, dim), 
                                values=self.df[dim]
                            ) for dim in dimensions
                        ]
                    )
                )
            else:
                # Create with color mapping
                # Make sure the color values are numeric and normalized
                color_values = self.df[color_var].values
                if pd.api.types.is_numeric_dtype(color_values):
                    # Normalize color values to 0-1 range for consistent coloring
                    color_min = np.min(color_values)
                    color_max = np.max(color_values)

                    # Only normalize if there's a range
                    if color_min != color_max:
                        normalized_colors = (color_values - color_min) / (color_max - color_min)
                    else:
                        normalized_colors = np.zeros_like(color_values)

                    fig = go.Figure(data=
                        go.Parcoords(
                            line=dict(
                                color=normalized_colors,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title=self.tooltips.get(color_var, color_var),
                                    tickvals=[0, 0.5, 1],
                                    ticktext=[f"{color_min:.2f}", 
                                              f"{(color_min + color_max)/2:.2f}", 
                                              f"{color_max:.2f}"]
                                )
                            ),
                            dimensions=[
                                dict(
                                    range=[self.df[dim].min(), self.df[dim].max()],
                                    label=self.tooltips.get(dim, dim), 
                                    values=self.df[dim]
                                ) for dim in dimensions
                            ]
                        )
                    )
                else:
                    # For non-numeric color variables, fall back to basic plot
                    print(f"Warning: Color variable '{color_var}' is not numeric. Using default coloring.")
                    fig = go.Figure(data=
                        go.Parcoords(
                            line=dict(color='blue', colorscale='Blues', 
                                     showscale=False),
                            dimensions=[
                                dict(
                                    range=[self.df[dim].min(), self.df[dim].max()],
                                    label=self.tooltips.get(dim, dim), 
                                    values=self.df[dim]
                                ) for dim in dimensions
                            ]
                        )
                    )

            # Adjust axis titles
            fig.update_layout(
                title=title,
                autosize=True, 
                margin=dict(l=80, r=80, t=50, b=50),
                font=dict(size=12),
                template=self.current_theme
            )
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating parallel coordinates plot: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            # Print error for debugging
            print(f"Parallel plot error: {str(e)}")
            import traceback
            traceback.print_exc()

        return fig


    def create_heatmap_figure(self, variables, title="Correlation Heatmap", method="pearson", colorscale="RdBu_r"):
        """Create an enhanced correlation heatmap with multiple correlation methods"""
        if len(variables) < 2:
            # Handle insufficient variables case
            fig = go.Figure()
            fig.add_annotation(
                text="Please select at least two variables for correlation analysis.",
                showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

        try:
            # Calculate correlation matrix with specified method
            corr = self.df[list(variables)].corr(method=method)

            # Create heatmap
            fig = px.imshow(
                corr, 
                text_auto='.2f',  # Format to 2 decimal places
                aspect="auto", 
                title=f"{title} ({method.capitalize()} Method)",
                color_continuous_scale=colorscale,
                zmin=-1,
                zmax=1,
                template=self.current_theme
            )

            # Improved layout with better text visibility
            fig.update_layout(
                autosize=True, 
                margin=dict(l=50, r=50, t=80, b=10),
                coloraxis_colorbar=dict(
                    title="Correlation",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1 Negative", "-0.5", "0 No Corr", "0.50", "<br>1 Positive"]
                )
            )

            # -- Works, just needs moved around --
            # Add annotations explaining the correlation value
            # fig.add_annotation(
            #     x=1.15,
            #     y=-0.5,
            #     xref="paper",
            #     yref="paper",
            #     text="<b>Correlation Guide:</b><br>"+
            #          "1.0: Perfect positive<br>"+
            #          "0.7-0.9: Strong positive<br>"+
            #          "0.4-0.6: Moderate positive<br>"+
            #          "0.1-0.3: Weak positive<br>"+
            #          "0: No correlation<br>"+
            #          "-0.1 to -0.3: Weak negative<br>"+
            #          "-0.4 to -0.6: Moderate negative<br>"+
            #          "-0.7 to -0.9: Strong negative<br>"+
            #          "-1.0: Perfect negative",
            #     showarrow=False,
            #     bgcolor="rgba(255, 255, 255, 0.7)",
            #     bordercolor="black",
            #     borderwidth=1,
            #     font=dict(size=10)
            # )

            # Plotly download button works now so not needed 
            # Add download button
            # fig.update_layout(
            #     updatemenus=[
            #         dict(
            #             type="buttons",
            #             direction="left",
            #             buttons=[
            #                 dict(
            #                     args=["toImage", {"format": "png", "width": 800, "height": 600}],
            #                     label="Download PNG",
            #                     method="relayout"
            #                 )
            #             ],
            #             pad={"r": 10, "t": 10},
            #             showactive=False,
            #             x=0.25,
            #             xanchor="left",
            #             y=1.3,
            #             yanchor="top"
            #         )
            #     ]
            # )

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating correlation heatmap: {str(e)}",
                showarrow=False,
                font=dict(size=14, color="red")
            )

        return fig


    def get_color_scale(self, ratio, base_color, min_color):
        """Generate a color between min_color and base_color based on ratio (0-1)"""
        # Simple linear interpolation between two hex colors
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(
                max(0, min(255, int(rgb[0]))),
                max(0, min(255, int(rgb[1]))),
                max(0, min(255, int(rgb[2])))
            )

        rgb1 = hex_to_rgb(min_color)
        rgb2 = hex_to_rgb(base_color)

        # Linear interpolation
        rgb_result = tuple(int(rgb1[i] + (rgb2[i] - rgb1[i]) * ratio) for i in range(3))
        return rgb_to_hex(rgb_result)
  

    def update_scatter(self, change):
        """Update 2D scatter plot with current settings"""
        try:
            new_fig = self.create_scatter_figure(
                self.scatter_x_dropdown.value,
                self.scatter_y_dropdown.value,
                self.scatter_color_dropdown.value,
                self.scatter_size_dropdown.value,
                self.scatter_trend.value
            )
            self.update_figurewidget(self.scatter_fig, new_fig)
        except Exception as e:
            print(f"Error updating scatter plot: {str(e)}")

    # update 3D scatter
    def update_scatter3d(self, change):
        """Update 3D scatter plot with current settings"""
        try:
            # Print debugging info
            # print(f"3D Scatter update - size dropdown value: {self.scatter3d_size_dropdown.value}")

            # Get size value - ensure it's a string, not a tuple or other type
            size_value = self.scatter3d_size_dropdown.value

            # Create new figure with proper parameters
            new_fig = self.create_scatter3d_figure(
                self.scatter3d_x_dropdown.value,
                self.scatter3d_y_dropdown.value,
                self.scatter3d_z_dropdown.value,
                self.scatter3d_color_dropdown.value,
                size_value
            )

            # Update the figure widget
            self.update_figurewidget(self.scatter3d_fig, new_fig)

        except Exception as e:
            print(f"Error updating 3D scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error trace

    def update_surface(self, change):
        """Update 3D surface plot with current settings"""
        try:
            new_fig = self.create_surface_figure(
                self.surface_x_dropdown.value,
                self.surface_y_dropdown.value,
                self.surface_z_dropdown.value,
                self.surface_colorscale.value
            )
            self.update_figurewidget(self.surface_fig, new_fig)
        except Exception as e:
            print(f"Error updating surface plot: {str(e)}")

    def update_parallel(self, _):
        """Update parallel coordinates plot with current settings"""
        try:
            dims = list(self.parallel_vars.value)
            if not dims:
                print("Select at least one variable for parallel coordinates.")
                return

            new_fig = self.create_parallel_figure(
                dims, 
                title=" ",
                color_var=self.parallel_color.value
            )
            self.update_figurewidget(self.parallel_fig, new_fig)
        except Exception as e:
            print(f"Error updating parallel coordinates plot: {str(e)}")

    def update_heatmap(self, _):
        """Update correlation heatmap with current settings"""
        try:
            vars_sel = list(self.heatmap_vars.value)
            if len(vars_sel) < 2:
                print("Select at least two variables for the heatmap.")
                return

            new_fig = self.create_heatmap_figure(
                vars_sel, 
                title="Correlation Heatmap",
                method=self.heatmap_method.value,
                colorscale=self.heatmap_colorscale.value
            )
            self.update_figurewidget(self.heatmap_fig, new_fig)
        except Exception as e:
            print(f"Error updating correlation heatmap: {str(e)}")

    def update_table(self, _):
        """Update the data table with current settings and filtering"""
        with self.table_output:
            clear_output(wait=True) #update without generating new block
            try:
                # Apply filtering if specified
                if self.table_filter_col.value != "None" and self.table_filter_text.value:
                    col = self.table_filter_col.value
                    val = self.table_filter_text.value

                    # Try to convert to numeric if possible for comparison
                    try:
                        numeric_val = float(val)
                        filtered_df = self.df[self.df[col] == numeric_val]
                    except ValueError:
                        # Use string contains for text fields
                        filtered_df = self.df[self.df[col].astype(str).str.contains(val, case=False, na=False)]
                else:
                    filtered_df = self.df

                # Get number of rows to display
                n_rows = min(self.table_rows.value, len(filtered_df))

                # Create styled dataframe
                styled_df = filtered_df.head(n_rows).style.set_caption(
                    f"Showing {n_rows} of {len(filtered_df)} rows " + 
                    f"(filtered from {len(self.df)} total rows)" if len(filtered_df) < len(self.df) else ""
                )

                # Apply styling
                styled_df = styled_df.highlight_max(color='lightgreen', axis=0, subset=self.numeric_columns)
                styled_df = styled_df.highlight_min(color='lightcoral', axis=0, subset=self.numeric_columns)

                display(styled_df)
            except Exception as e:
                print(f"Error updating table: {str(e)}")

    def update_statistics(self, _):
        """Calculate and display summary statistics for selected variables"""
        with self.stats_output:
            clear_output(wait=True) #update without generating new block
            try:
                # Get selected variables
                vars_sel = list(self.stats_vars.value)

                if not vars_sel:
                    print("Please select at least one variable for statistics.")
                    return

                # Basic statistics for selected variables
                if self.stats_categorical.value == "None":
                    # Regular statistics for all data
                    stats_df = self.df[vars_sel].describe().T

                    # Add additional statistics
                    stats_df['median'] = self.df[vars_sel].median()
                    stats_df['skew'] = self.df[vars_sel].skew()
                    stats_df['kurtosis'] = self.df[vars_sel].kurtosis()
                    stats_df['missing'] = self.df[vars_sel].isna().sum()
                    stats_df['missing_pct'] = self.df[vars_sel].isna().mean() * 100

                    # Reorder columns
                    cols_order = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 
                                 'skew', 'kurtosis', 'missing', 'missing_pct']
                    stats_df = stats_df[cols_order]

                    # Style the dataframe
                    styled_stats = stats_df.style.format({
                        'mean': '{:.4f}',
                        'median': '{:.4f}',
                        'std': '{:.4f}',
                        'min': '{:.4f}',
                        '25%': '{:.4f}',
                        '50%': '{:.4f}',
                        '75%': '{:.4f}',
                        'max': '{:.4f}',
                        'skew': '{:.4f}',
                        'kurtosis': '{:.4f}',
                        'missing_pct': '{:.2f}%'
                    })

                    display(styled_stats)

                    # Display histogram summary for each variable
                    print("\n\nHistogram Summary:")
                    for var in vars_sel:
                        fig = px.histogram(
                            self.df, 
                            x=var, 
                            title=f"Distribution of {var}",
                            template=self.current_theme
                        )

                        # Add mean and median lines
                        mean_val = self.df[var].mean()
                        median_val = self.df[var].median()

                        fig.add_vline(x=mean_val, line_dash="solid", line_color="red",
                                      annotation_text=f"Mean: {mean_val:.4f}")
                        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                                      annotation_text=f"Median: {median_val:.4f}")

                        display(fig)
                else:
                    # Group by the selected categorical variable
                    cat_var = self.stats_categorical.value

                    # Calculate group statistics
                    group_stats = []
                    for var in vars_sel:
                        # Basic grouped statistics
                        group_data = self.df.groupby(cat_var)[var].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).reset_index()

                        # Add variable name
                        group_data['variable'] = var

                        # Reorder columns
                        group_data = group_data[['variable', cat_var, 'count', 'mean', 'median', 'std', 'min', 'max']]

                        group_stats.append(group_data)

                    # Combine all variables
                    if group_stats:
                        all_stats = pd.concat(group_stats, ignore_index=True)

                        # Style and display
                        styled_group_stats = all_stats.style.format({
                            'mean': '{:.4f}',
                            'median': '{:.4f}',
                            'std': '{:.4f}',
                            'min': '{:.4f}',
                            'max': '{:.4f}'
                        })

                        display(styled_group_stats)

                        # Create group comparison plots
                        for var in vars_sel:
                            # Box plot by group
                            box_fig = px.box(
                                self.df, 
                                x=cat_var, 
                                y=var, 
                                title=f"Distribution of {var} by {cat_var}",
                                template=self.current_theme
                            )
                            display(box_fig)

                            # Bar chart of means by group
                            bar_fig = px.bar(
                                self.df.groupby(cat_var)[var].mean().reset_index(),
                                x=cat_var,
                                y=var,
                                title=f"Mean {var} by {cat_var}",
                                template=self.current_theme
                            )

                            # Add error bars showing standard deviation
                            error_y = self.df.groupby(cat_var)[var].std().values
                            bar_fig.update_traces(error_y=error_y)

                            display(bar_fig)
                    else:
                        print(f"No valid grouped statistics available for {cat_var} and {vars_sel}")

            except Exception as e:
                print(f"Error calculating statistics: {str(e)}")
                import traceback
                traceback.print_exc()

    def display(self):
        """Display the dashboard"""
        display(self.main_layout)

    def create_data_table(self, rows=10, filter_col="None", filter_value=""):
        """Create an interactive data table with filtering"""
        try:
            # Apply filtering if specified
            if filter_col != "None" and filter_value:
                # Try to convert to numeric if possible for comparison
                try:
                    numeric_val = float(filter_value)
                    filtered_df = self.df[self.df[filter_col] == numeric_val]
                except ValueError:
                    # Use string contains for text fields
                    filtered_df = self.df[self.df[filter_col].astype(str).str.contains(filter_value, case=False, na=False)]
            else:
                filtered_df = self.df

            # Get number of rows to display
            n_rows = min(rows, len(filtered_df))

            # Create styled dataframe
            styled_df = filtered_df.head(n_rows).style.set_caption(
                f"Showing {n_rows} of {len(filtered_df)} rows " + 
                f"(filtered from {len(self.df)} total rows)" if len(filtered_df) < len(self.df) else ""
            )

            # Apply styling
            styled_df = styled_df.highlight_max(color='lightgreen', axis=0, subset=self.numeric_columns)
            styled_df = styled_df.highlight_min(color='lightcoral', axis=0, subset=self.numeric_columns)

            return styled_df

        except Exception as e:
            print(f"Error creating data table: {str(e)}")
            return pd.DataFrame({"Error": [f"Failed to create table: {str(e)}"]})

    def calculate_statistics(self, variables, group_by="None"):
        """Calculate statistics for the selected variables, optionally grouped"""
        if not variables:
            return pd.DataFrame({"Error": ["No variables selected for statistics"]})

        try:
            if group_by == "None":
                # Basic statistics for selected variables
                stats_df = self.df[variables].describe().T

                # Add additional statistics
                stats_df['median'] = self.df[variables].median()
                stats_df['skew'] = self.df[variables].skew()
                stats_df['kurtosis'] = self.df[variables].kurtosis()
                stats_df['missing'] = self.df[variables].isna().sum()
                stats_df['missing_pct'] = self.df[variables].isna().mean() * 100

                # Reorder columns
                cols_order = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 
                             'skew', 'kurtosis', 'missing', 'missing_pct']
                stats_df = stats_df[cols_order]

                return stats_df
            else:
                # Group by the selected categorical variable
                result_dfs = []

                for var in variables:
                    group_stats = self.df.groupby(group_by)[var].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).reset_index()

                    group_stats['variable'] = var
                    result_dfs.append(group_stats)

                if result_dfs:
                    return pd.concat(result_dfs, ignore_index=True)
                else:
                    return pd.DataFrame({"Error": ["No valid statistics calculated"]})

        except Exception as e:
            return pd.DataFrame({"Error": [f"Statistics calculation failed: {str(e)}"]})

    def create_histogram(self, variable, bins=20):
        """Create a histogram for a numeric variable"""
        if variable not in self.df.columns or variable not in self.numeric_columns:
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {variable} is not a valid numeric column", showarrow=False)
            return fig

        fig = px.histogram(
            self.df, 
            x=variable, 
            nbins=bins,
            title=f"Distribution of {variable}",
            template=self.current_theme
        )

        # Add mean and median lines
        mean_val = self.df[variable].mean()
        median_val = self.df[variable].median()

        fig.add_vline(x=mean_val, line_dash="solid", line_color="red",
                      annotation_text=f"Mean: {mean_val:.4f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                      annotation_text=f"Median: {median_val:.4f}",
                      annotation_position="bottom left")

        fig.update_layout(
            xaxis_title=self.tooltips.get(variable, variable),
            yaxis_title="Count",
            autosize=True,
            margin=dict(l=50, r=50, t=70, b=50)
        )

        return fig