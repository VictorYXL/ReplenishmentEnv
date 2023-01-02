import os
import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line, Timeline, Grid
from datetime import datetime, timedelta
from ReplenishmentEnv.env.agent_states import AgentStates

"""
use pyecharts to show each sku's policy and overall visualization.
"""
class Visualizer:
    def __init__(
        self, 
        agent_states: AgentStates,
        reward_info_list:str,
        start_date: datetime, 
        sku_list: list, 
        facility_list: list, 
        state_items: list,
        output_dir:str,
        warmup_length=0
    ):
        self.agent_states       = agent_states
        self.reward_info_list   = reward_info_list
        self.sku_list           = sku_list
        self.facility_list      = facility_list
        self.start_date         = start_date
        self.state_items        = state_items
        self.output_dir         = output_dir
        self.warmup_length      = warmup_length
        self.colors             = ["blue", "orange", "green", "aqua", "yellow", "red", "black"]
        self.facility_to_id     = self.agent_states.facility_to_id
        self.sku_to_id          = self.agent_states.sku_to_id
    
    def render_single(self, facility, sku="overview"):
        periods = self.agent_states.current_step
        dates = pd.date_range(self.start_date, periods=periods)

        tl = Timeline(init_opts=opts.InitOpts(width="1500px", height="800px"))
        tl.add_schema(pos_bottom="bottom", is_auto_play=False, \
            label_opts = opts.LabelOpts(is_show=True, position="bottom"))

        state_line = Line().add_xaxis(xaxis_data = dates.tolist())
        for index, item in enumerate(self.state_items):
            if sku == "overview":
                value = np.sum(self.agent_states[facility, item, "history", "all_skus"][self.warmup_length:], 1)
            else:
                value = self.agent_states[facility, item, "history", sku][self.warmup_length:]
            value = value.tolist()
            color = self.colors[index % len(self.colors)]
            state_line.add_yaxis(
                series_name=item,
                y_axis=value,
                symbol_size=8,
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False, color=color),
                linestyle_opts=opts.LineStyleOpts(width=1.5, color=color),
                is_smooth=True,
                itemstyle_opts=opts.ItemStyleOpts(color=color),
            )
            state_line.set_global_opts(
                title_opts=opts.TitleOpts(title="SKU States", pos_top="top", pos_left='left', pos_right='left'),
                xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                legend_opts=opts.LegendOpts(pos_left="center"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                datazoom_opts=[
                    opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                    opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                ],
            )
        
        reward_line = Line().add_xaxis(xaxis_data = dates.tolist())
        
        facility_id  = self.facility_to_id[facility]
        reward_items = list(self.reward_info_list[0].keys())
        for index, item in enumerate(reward_items): 
            if sku == "overview":
                value = [np.sum(reward_info[item][facility_id]) for reward_info in self.reward_info_list]
            else:
                sku_id = self.sku_to_id[sku]
                value = [reward_info[item][facility_id][sku_id] for reward_info in self.reward_info_list]
            value = value[self.warmup_length:]
            color = self.colors[index % len(self.colors)]
            reward_line.add_yaxis(
                series_name=item,
                y_axis=value,
                symbol_size=8,
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False, color=color),
                linestyle_opts=opts.LineStyleOpts(width=1.5, color=color),
                is_smooth=True,
                itemstyle_opts=opts.ItemStyleOpts(color=color),
            )
        reward_line.set_global_opts(
            title_opts=opts.TitleOpts(title="Reward States", pos_top='45%', pos_left='left', pos_right='left'),
            xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
            yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top='45%'),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            datazoom_opts=[
                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
            ],
        )

        grid = (
            Grid()
            .add(state_line, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, height="30%"))
            .add(reward_line, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, pos_top="50%", height="30%"))
        )
            
        tl.add(grid, "{}".format(self.start_date.strftime("%Y-%m-%d")))
        output_name = "{}_{}.html".format(facility, sku)
        tl.render(os.path.join(self.output_dir, output_name))

    def render(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for facility in self.facility_list:
            for sku in self.sku_list:
                self.render_single(facility, sku)
            self.render_single(facility, "overview")


    
