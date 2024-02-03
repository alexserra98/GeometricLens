from metrics.utils import class_imbalance
class DataFrameQuery:
    def __init__(self, query: dict, post_process_query=None):
        self.query = query
        self.post_process_query = post_process_query

    def _build_query(self, parameter, argument):

        if isinstance(argument, list):
            return f"{parameter} in {argument}"
        else:  # assuming it's a string
            return f"{parameter} == '{argument}'"
        

    def apply_query(self, dataframe):
        query_string = self.query_string()
        if query_string:
            dataframe=dataframe.query(query_string)
        if self.post_process_query:
            # currently resolving class imbalance is the only post processing step
            dataframe = class_imbalance(dataframe, self.post_process_query["balanced"])
        return dataframe

    def _combine_queries(self, query_list):
        queries = [q for q in query_list if q is not None]
        return " and ".join(queries)
    def query_string(self):
        query_list = [self._build_query(parameter, argument) for parameter, argument in self.query.items()]
        query_string = self._combine_queries(query_list)
        return query_string

# Example usage
# df = your pandas dataframe
# query = DataFrameQuery('dataset_name', ['model1', 'model2'])
# filtered_df = query.apply_query(df)
