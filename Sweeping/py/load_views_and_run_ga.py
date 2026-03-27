# Old piece of code not deleted in case useful in the future to show the history of an individual.
if False:
    if RETURN_HISTORY:
        graph = networkx.DiGraph(history.genealogy_tree)
        graph = graph.reverse()  # Make the graph top-down
        colors = []
        for i in graph:
            individual = history.genealogy_history[i]
            fit = individual.fitness.getValues()
            if len(fit) == 0:
                colors.append(0.)
            else:
                colors.append(fit[0])  # Just the first objective is used
        pos = graphviz_layout(graph, prog='dot')
        # print(str(colors))
        networkx.draw(graph, pos=pos, node_color=colors)
        plt.savefig("./prad/genealogy.png", bbox_inches='tight', dpi=600)
        plt.close()
