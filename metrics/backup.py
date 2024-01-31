class ShotMetrics(Metrics):
  """
  Class to compute the metrics of a run. It takes a list of request results computed by generate.py
  and compute the per instance metrics 
  """
  def __init__(self, scenario_result: ScenarioResult):
    self.requests_results = scenario_result.requests_results
    self.train_instances = scenario_result.train_instances
    self.dataset = scenario_result.dataset
    self.model = scenario_result.model_name
    self.basic_metric, self.hidden_states = self.set_dataframes()

  def set_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate in two different dataframes the data for basic metrics and the hidden states of all instances
    Output
    ----------
    basic_metrics: pd.DataFrame(num_instances, num_metrics)
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    basic_metrics_dict = {"loss":[], "perplexity":[], "std_exact_match":[],"std_quasi_exact_match":[], "ref_exact_match":[]}
    hidden_states_dict = {"hidden_states": [],"layer": [], "match": [], "answered_letter": [], "gold_letter": []}
    for request_result in self.requests_results:
      basic_metrics_dict["loss"].append(request_result.loss)
      basic_metrics_dict["perplexity"].append(np.exp(request_result.loss))
      basic_metrics_dict["std_exact_match"].append(exact_match(request_result.preds["std_pred"]["letter"], request_result.gold["letter"]))
      basic_metrics_dict["ref_exact_match"].append(exact_match(request_result.preds["only_ref_pred"]["letter"],request_result.gold["letter"]))
      basic_metrics_dict["std_quasi_exact_match"].append(quasi_exact_match(request_result.preds["std_pred"]["letter"], request_result.gold["letter"]))
      for layer in ["last","sum"]:
        hidden_states_dict["hidden_states"].append(request_result.hidden_states[layer])
        hidden_states_dict["layer"].append(layer)
        match = "correct" if basic_metrics_dict["std_exact_match"][-1] else "wrong"
        hidden_states_dict["match"].append(match)
        hidden_states_dict["answered_letter"].append(request_result.preds["only_ref_pred"]["letter"])
        hidden_states_dict["gold_letter"].append(request_result.gold["letter"])
    basic_metrics = pd.DataFrame(basic_metrics_dict)
    hidden_states = pd.DataFrame(hidden_states_dict)
    return basic_metrics, hidden_states
  

  
  def compute_nn(self,k = 20) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the nearest neighbours of each instance in the run per layer
    using the provided methodv
    Output
    ----------
    Dict[k-method, Array(num_layers, num_instances, k_neighbours)]
    """
    hidden_states = HiddenStates(self.hidden_states)
    warn(f'Computing nearest neighbours with k={k}')
    return hidden_states.get_nearest_neighbour(k) 
  
  #TODO use property decorator
  def basic_metric_mean(self) -> Dict[str, float]:
    output_dict = {column_name: self.basic_metric[column_name].mean() for column_name in self.basic_metric.columns}
    return output_dict  



class Overlap(ABC):
  """
  Abstract Class for compute different kinds of overlap 
  """  
  def __init__(self, db: MetadataDB, query: DataFrameQuery, path_result: Path) -> None:
    self.db = db
    self.query = query
    self.df = self.set_dataframes()
    self.path_result = path_result
  
  @abstractmethod
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    pass
  
  @abstractmethod
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    pass
  
class LabelOverlap(Overlap):
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df = pd.read_sql("SELECT * FROM metadata", self.db.conn)
    columns = ['id_instance', 
               'dataset', 
               'train_instances', 
               'model_name',
               'only_ref_pred', 
               'method']
    df = df[columns]  # Keep only the columns in the list "columns"
    return df
  def _compute_overlap(self, label) -> Dict[str, Dict[str, np.ndarray]]:
    df = df.rename(columns={'dataset': 'subject'})
    hidden_states = HiddenStates(self.df, self.path_result)
    return hidden_states.layer_overlap_label(label)

class SubjectOverlap(LabelOverlap):
  
  def set_dataframes(self) -> pd.DataFrame:
    """
    Aggregate in a dataframe the hidden states of all instances
    ----------
    hidden_states: pd.DataFrame(num_instances, num_layers, model_dim)
    """
    df = super().set_dataframes()
    df = self.query.apply_query(df)
    df = df.rename(columns={'dataset': 'subject'})
    df["train_instances"] = df["train_instances"].astype(str)
    return df 
  
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    return self._compute_overlap("subject")


  
class LetterOverlap(LabelOverlap):
  def set_dataframes(self) -> DataFrame:
    df= super().set_dataframes()
    df = self.query.apply_query(df)
    df["train_instances"] = df["train_instances"].astype(str)
    return df 
  
  def compute_overlap(self) -> Dict[str, Dict[str, np.ndarray]]:
    return self._compute_overlap("letter")
