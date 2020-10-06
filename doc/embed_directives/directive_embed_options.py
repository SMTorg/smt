from sphinx_auto_embed.directive import Directive


class DirectiveEmbedOptions(Directive):
    """
    Directive for embedding a table from an OptionsDictionary instance.

    The 3 arguments are the module name, class name, and attribute name.
    """

    NAME = "embed-options-table"
    NUM_ARGS = 3

    def run(self, file_dir, file_name, embed_num_indent, args):
        module_path, class_name, attribute_name = args

        exec("from {} import {}".format(module_path, class_name), globals())
        exec("obj = {}()".format(class_name), globals())

        options = getattr(obj, attribute_name)

        outputs = []
        for option_name, option_data in options._declared_entries.items():
            name = option_name
            default = option_data["default"]
            values = option_data["values"]
            types = option_data["types"]
            desc = option_data["desc"]

            if types is not None:
                if not isinstance(types, (tuple, list)):
                    types = (types,)

                types = [type_.__name__ for type_ in types]

            if values is not None:
                if not isinstance(values, (tuple, list)):
                    values = (values,)

                values = [value for value in values]

            outputs.append([name, default, values, types, desc])

        lines = []
        lines.append(" " * embed_num_indent + ".. list-table:: List of options\n")
        lines.append(" " * embed_num_indent + " " * 2 + ":header-rows: 1\n")
        lines.append(" " * embed_num_indent + " " * 2 + ":widths: 15, 10, 20, 20, 30\n")
        lines.append(" " * embed_num_indent + " " * 2 + ":stub-columns: 0\n")
        lines.append("\n")
        lines.append(" " * embed_num_indent + " " * 2 + "*  -  Option\n")
        lines.append(" " * embed_num_indent + " " * 2 + "   -  Default\n")
        lines.append(" " * embed_num_indent + " " * 2 + "   -  Acceptable values\n")
        lines.append(" " * embed_num_indent + " " * 2 + "   -  Acceptable types\n")
        lines.append(" " * embed_num_indent + " " * 2 + "   -  Description\n")

        for output in outputs:
            for entry in [output[0]]:
                lines.append(" " * embed_num_indent + " " * 2 + "*  -  %s\n" % entry)
            for entry in output[1:]:
                lines.append(" " * embed_num_indent + " " * 2 + "   -  %s\n" % entry)

        return lines
