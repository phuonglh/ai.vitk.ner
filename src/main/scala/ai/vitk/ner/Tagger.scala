package ai.vitk.ner

import java.io.{File, FileInputStream}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer


/**
 * @author Phuong LE-HONG
 * <p>
 * Sep 5, 2016, 12:22:53 PM
 * <p>
 * Implementation of a sequence tagger.
 */
object Tagger {
  val logger = LoggerFactory.getLogger(Tagger.getClass)
  var verbose = false
  var numFeatures = math.pow(2, 15).toInt
  var iterations = 600
  
  def setVerbose(verbose: Boolean): Unit = this.verbose = verbose
  def setDimension(numFeatures: Int): Unit = this.numFeatures = numFeatures 
  def setIterations(iterations: Int): Unit = this.iterations = iterations
  
  /**
   * Trains a MLR model: (sentences, model parameters) => model. 
   * @param spark Spark session
   * @param sentences list of training sentences
   * @param modelPath the path to save model to
   * @param numFeatures number of features used in feature hashing (domain dimension)
   * @param isReversed train a backward model or forward model
   * @return a pipeline model.
   */
  def train(spark: SparkSession, sentences: List[Sentence], modelPath: String, numFeatures: Int, isReversed: Boolean): PipelineModel = {
    logger.info("Preparing data frame for training... Please wait.")
    val trainingSentences = if (!isReversed) {
      sentences
    } else {
      sentences.map(s => Sentence(s.tokens.reverse))
    }
    val df = createDF(spark, trainingSentences)
    df.cache() 
    
    // create and fit a processing pipeline 
    val labelIndexer = new StringIndexer().setInputCol("tag").setOutputCol("label")
    val tokenizer = new Tokenizer().setInputCol("bof").setOutputCol("tokens")
    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("features").setBinary(true).setNumFeatures(numFeatures)
    val mlr = new LogisticRegression().setMaxIter(iterations).setRegParam(1E-6).setStandardization(false)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, hashingTF, mlr))
    val model = pipeline.fit(df)
    
    // overwrite the trained pipeline
    model.write.overwrite().save(if (isReversed) modelPath + "-reversed" else modelPath)
    // print some strings to debug the model
    if (verbose) {
      logger.info("#(contexts) = " + df.count())
      df.show(10, false)
      val labels = model.stages(0).asInstanceOf[StringIndexerModel].labels
      logger.info("#(labels) = " + labels.length)
      val logreg = model.stages(3).asInstanceOf[LogisticRegressionModel]
      logger.info(logreg.explainParams())
    }
    model 
  }
  
  /**
   * Tags a list of sentences: (model, input sentences) => output sentences.
   * @param spark
   * @param modelPath
   * @param sentences
   * @return a list of tagged sentences.
   */
  def tag(spark: SparkSession, modelPath: String, sentences: List[Sentence]): List[Sentence] = {
    val model = PipelineModel.load(modelPath)
    val decoder = new Decoder(spark, DecoderType.Greedy, model)
    decoder.decode(sentences)
  }

  /**
   * Tags a list of sentences and saves the result to an output file in a two-column format, which  
   * is suitable for the evaluation tool 'conlleval' of the CoNLL-2003 NER shared-task. 
   * @param spark
   * @param modelPath
   * @param sentences
   * @param outputPath
   * @param isReversed
   */
  def tag(spark: SparkSession, modelPath: String, sentences: List[Sentence], outputPath: String, isReversed: Boolean) {
    val lines = new ListBuffer[String]()
    val sents = if (!isReversed) sentences else { 
      sentences.map(s => Sentence(s.tokens.reverse))
    } 
    // copy sents to xs 
    val xs = sents.map(x => Sentence(x.tokens.clone()))
    // tag sents; their annotation will be updated during the process
    val ys = tag(spark, modelPath, sents)
    // prepare results of the format (correct tag, predicted tag)
    for (i <- 0 until sents.length) {
      val x = xs(i)
      val y = ys(i)
      val line = ListBuffer[String]()
      for (j <- 0 until x.length) {
        line.append(x.tokens(j).annotation(Label.NamedEntity) + ' ' + y.tokens(j).annotation(Label.NamedEntity))
      }
      if (isReversed)
        lines.append(line.reverse.mkString("\n"))
      else lines.append(line.mkString("\n"))
      lines.append("\n\n")
    }
    // save the lines
    val pw = new java.io.PrintWriter(new java.io.File(outputPath))
    try {
      lines.foreach(line => pw.write(line))
    } finally {
      pw.close()
    }
  }
  
  private def createDF(spark: SparkSession, sentences: List[Sentence]): Dataset[LabeledContext] = {
    val contexts = sentences.flatMap {
      sentence => Featurizer.extract(sentence)
    }
    import spark.implicits._
    spark.createDataFrame(contexts).as[LabeledContext]
  }

  /**
   * Trains a machine learning model to do named-entity recognition.
   * @param spark
   * @param language
   * @param isReversed
   */
  def trainNER(spark: SparkSession, language: Language.Value, isReversed: Boolean) {
    logger.info("Loading training corpus for " + language + " NER...")
    val corpusPack = new CorpusPack(language)
    val trainingSet = CorpusReader.readCoNLL(corpusPack.resourcePaths._1)
    logger.info("#(dimension) = " + numFeatures)
    logger.info("#(training sentences) = " + trainingSet.length)
    val modelPath = "src/main/resources" + corpusPack.modelPath
    train(spark, trainingSet, modelPath + "mlr", numFeatures, isReversed)
  }

  /**
   * Tests a machine learning model for NER on test sets.
   * @param spark
   * @param language
   * @param isReversed
   */
  def testNER(spark: SparkSession, language: Language.Value, isReversed: Boolean) {
    val corpusPack = new CorpusPack(language)
    val testSet = CorpusReader.readCoNLL(corpusPack.resourcePaths._2)
    logger.info("#(test examples) = " + testSet.length)
    val modelPathPrefix = "src/main/resources" + corpusPack.modelPath
    val suffix = if (isReversed) "-reversed" else ""
    val modelPath = modelPathPrefix + "mlr" + suffix
    val outputPath = "src/main/resources" + corpusPack.resourcePaths._2 + ".out" +  suffix
    tag(spark, modelPath, testSet, outputPath, isReversed)  
  }

  private def combine(spark: SparkSession, modelPath: String, sentences: List[Sentence], outputPath: String) {
    // forward tagging 
    val us = sentences.map(x => Sentence(x.tokens.clone()))
    val ys = tag(spark, modelPath, us)
    
    // backward tagging
    val vs = sentences.map(x => Sentence(x.tokens.clone().reverse))
    val zs = tag(spark, modelPath + "-reversed", vs)
    
    // combine the result of ys and zs 
    val lines = ListBuffer[String]()
    for (i <- 0 until ys.length) {
      val y = ys(i)
      val z = Sentence(zs(i).tokens.reverse)
      val s = ListBuffer[Token]()
      for (j <- 0 until y.length)
        if (z.tokens(j).namedEntity.endsWith("LOC") && !y.tokens(j).namedEntity.endsWith("ORG")) 
          s.append(z.tokens(j)) else s.append(y.tokens(j))

      val pair = sentences(i).tokens zip s
      val line = pair.map(p => p._1.namedEntity + ' ' + p._2.namedEntity).mkString("\n")
      lines.append(line)
      lines.append("\n\n")
    }
    
    // save the lines
    val pw = new java.io.PrintWriter(new java.io.File(outputPath))
    try {
      lines.foreach(line => pw.write(line))
    } finally {
      pw.close()
    }
    
  }

  /**
    * Tags an input file by combining both forward and backward models.
    * @param spark
    * @param language
    * @param input an input file in CoNLL-2003 format (VLSP format)
    */
  def combineNER(spark: SparkSession, language: Language.Value, input: String) = {
    val corpusPack = new CorpusPack(language)
    val inputStream = new FileInputStream(new File(input))
    val testSet = CorpusReader.readCoNLL(inputStream)
    logger.info("#(test examples) = " + testSet.length)
    val modelPathPrefix = "src/main/resources" + corpusPack.modelPath
    val modelName = "mlr"
    val modelPath = modelPathPrefix + modelName
    val outputPath = input + ".out"
    combine(spark, modelPath, testSet, outputPath)
  }
  
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val options = new Options(args)
    val spark = SparkSession.builder().appName(getClass.getName)
      .master(options.master()).config("spark.executor.memory", options.memory())
      .getOrCreate()
    val language = options.language() match {
      case "vi" => Language.Vietnamese
      case "en" => Language.English
      case "ja" => Language.Japanese
    }

    setDimension(options.dimension())
    setVerbose(options.verbose())
    setIterations(options.iteration())
    
    val mode = options.mode()
    mode match {
      case "train" => trainNER(spark, language, options.reversed())
      case "eval" => testNER(spark,  language, options.reversed())
      case "tag" => {
        val input = options.input()
        combineNER(spark, language, input)
      }
    }
  }
}

