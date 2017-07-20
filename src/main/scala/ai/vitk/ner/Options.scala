package ai.vitk.ner

import org.rogach.scallop.ScallopConf

/**
  * Created by phuonglh on 6/30/17.
  * 
  * Some options used in the tool.
  * 
  */
class Options(arguments: Seq[String]) extends ScallopConf(arguments) {
  val master = opt[String](default = Some("local[*]"), descr = "the Spark master URL")
  val memory = opt[String](default = Some("8g"), descr = "executor memory")
  val mode = opt[String](default = Some("tag"), descr = "mode of the tagger, either 'train', 'tag' or 'eval'")
  val verbose = opt[Boolean](default = Some(false), descr = "verbose mode")
  val language = opt[String](default = Some("vi"), descr = "natural language in use, either 'vi', 'en' or 'ja'")
  val dimension = opt[Int](default = Some(32768), descr = "domain dimension for feature hashing")
  val iteration = opt[Int](default = Some(600), descr = "max number of iterations in training")
  val independent = opt[Boolean](default = Some(false), descr = "use only independent features")
  val reversed = opt[Boolean](default = Some(false), descr = "backward model")
  val input = opt[String](default = Some("test.txt"), descr = "input file for tagging")
  verify()
}
