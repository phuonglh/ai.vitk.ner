package ai.vitk.ner

import java.io.{File, InputStream}

import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer
import scala.io.Source

object CorpusReader {
  val logger = LoggerFactory.getLogger(CorpusReader.getClass)

  /**
    * Reads a NER corpus in CoNLL-2003 format.
    * @param inputStream
    * @return a list of sentences.
    */
  def readCoNLL(inputStream: InputStream): List[Sentence] = {
    val lines = (Source.fromInputStream(inputStream).getLines() ++ List("")).toArray
    val sentences = new ListBuffer[Sentence]()
    val indices = lines.zipWithIndex.filter(p => p._1.trim.isEmpty).map(p => p._2)
    var u = 0
    var v = 0
    for (i <- (0 until indices.length)) {
      v = indices(i)
      if (v > u) { // don't treat two consecutive empty lines
        val s = lines.slice(u, v)
        val tokens = s.map(line => {
          val parts = line.trim.split("\\s+")
          Token(parts(0), Map(Label.PartOfSpeech -> parts(1), Label.Chunk -> parts(2), Label.NamedEntity -> parts(3)))
        })
        sentences.append(Sentence(tokens.toList.to[ListBuffer]))
      }
      u = v + 1
    }
    sentences.toList
  }
  
  /**
    * Reads a NER corpus in CoNLL-2003 format.
    * @param resourcePath
    * @return a list of sentences.
    */
  def readCoNLL(resourcePath: String): List[Sentence] = {
    val stream = getClass.getResourceAsStream(resourcePath)
    readCoNLL(stream)
  }
 
  /**
   * Reads a VLSP test file and builds sentences to tag.
   * @param resourcePath
   * @return a list of [[Sentence]]
   */
  def readVLSPTest1(resourcePath: String): List[Sentence] = {
    // read lines of the file and remove lines which contains "<s>"
    val stream = getClass.getResourceAsStream(resourcePath)
    val lines = Source.fromInputStream(stream).getLines().toList.filter {
      line => line.trim != "<s>"
    }
    val sentences = new ListBuffer[Sentence]()
    var tokens = new ListBuffer[Token]()
    for (i <- (0 until lines.length)) {
      val line = lines(i).trim
      if (line == "</s>") {
        if (!tokens.isEmpty) sentences.append(Sentence(tokens))
        tokens = new ListBuffer[Token]()
      } else {
        val parts = line.split("\\s+")
        if (parts.length < 3) 
          logger.error("Invalid line = " + line) 
        else 
          tokens.append(Token(parts(0), Map(Label.PartOfSpeech -> parts(1), Label.Chunk -> parts(2))))
      }
    }
    logger.info(resourcePath + ", number of sentences = " + sentences.length)
    sentences.toList
  }
  
  def readVLSPTest2(dir: String): List[Sentence] = {
    def getListOfFiles: List[File] = {
      val d = new File(dir)
      if (d.exists && d.isDirectory) {
        d.listFiles.filter(_.isFile).toList
      } else {
        List[File]()
      }
    }
    val files = getListOfFiles
    logger.info("Number of test files = " + files.length)
    files.flatMap {
      file  => {
        val x = file.getAbsolutePath
        val resourcePath = x.substring(x.indexOf("/ner"))
        readVLSPTest1(resourcePath)
      } 
    }
  }
  
  
  def main(args: Array[String]): Unit = {
    val path = "/ner/vi/train.txt"
    val sentences = readCoNLL(path)
    logger.info("Number of sentences = " + sentences.length)
    sentences.take(10).foreach(s => logger.info(s.toString))
    sentences.takeRight(10).foreach(s => logger.info(s.toString))
  }
}