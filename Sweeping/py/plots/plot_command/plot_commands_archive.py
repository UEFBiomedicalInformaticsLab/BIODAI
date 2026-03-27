from manu.mv_manu.mv_manu_batteries import SARC_MV_MANU_BATTERY, KIRC_MV_MANU_BATTERY, LGG_MV_MANU_BATTERY
from plots.plot_command.plot_command import PlotCommand

SARC_MV_COMMAND = PlotCommand(batteries=[SARC_MV_MANU_BATTERY])
KIRC_MV_COMMAND = PlotCommand(batteries=[KIRC_MV_MANU_BATTERY])
LGG_MV_COMMAND = PlotCommand(batteries=[LGG_MV_MANU_BATTERY])
MV_COMMANDS = [SARC_MV_COMMAND, KIRC_MV_COMMAND, LGG_MV_COMMAND]
