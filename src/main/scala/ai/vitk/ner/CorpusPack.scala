package ai.vitk.ner

/**
  * Created by phuonglh on 7/19/17.
  */
class CorpusPack(val language: Language.Value) {
  
  def modelPath: String = {
    language match {
      case Language.Vietnamese => "/ner/vi/"
      case Language.English => "/ner/en/"
      case Language.Japanese => "/ner/ja/"
    }
  }

  def resourcePaths: (String, String) = {
    language match {
      case Language.Vietnamese => ("/ner/vi/vlsp.txt", "/ner/vi/test.txt")
      case Language.English => ("/ner/en/eng.train", "/ner/en/eng.test")
      case Language.Japanese => ("", "")
    }
  }
}
