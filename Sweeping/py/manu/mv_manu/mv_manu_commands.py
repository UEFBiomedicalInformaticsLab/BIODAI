from plots.plot_command.plot_command import PlotCommand
from manu.mv_manu.mv_manu_batteries import KIRC_MV_MANU_BATTERY, LGG_MV_MANU_BATTERY, SARC_MV_MANU_BATTERY

SARC_MV_MANU_COMMAND = PlotCommand(batteries=[SARC_MV_MANU_BATTERY])
KIRC_MV_MANU_COMMAND = PlotCommand(batteries=[KIRC_MV_MANU_BATTERY])
LGG_MV_MANU_COMMAND = PlotCommand(batteries=[LGG_MV_MANU_BATTERY])
MV_MANU_COMMANDS = [SARC_MV_MANU_COMMAND, KIRC_MV_MANU_COMMAND, LGG_MV_MANU_COMMAND]