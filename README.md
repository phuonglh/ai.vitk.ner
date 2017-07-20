# Vietnamese Named Entity Recognition
---

This is the first public release of a Vietnamese named entity recognition system, developed by [Phuong
LE-HONG](http://mim.hus.vnu.edu.vn/phuonglh) at College of Sciences, Vietnam National University in Hanoi. 


This tool implements a conditional Markov model (CMM), a common probabilistic model for sequence labelling. In
essence, CMM is a discriminative model which models the conditional probability distribution `P(tag sequence|word sequence)`. 
This probability is decomposed into a chain of local probability distributions `P(tag|word)` by using the Markov property. 
Each local probability distribution is a log-linear model (also called a maximum entropy model). Actually, the implemented approaches combines both 
a forward sequence model and a backward sequence model and token regular expressions to achieve the first-rank result of 
the VLSP 2016 NER shared task, organized by the Vietnamese Speech and Language Processing Society. On the standard test set, 
its F1 score is about 88.5%.   

The detailed approach is described in the following paper:

* [Vietnamese Named Entity Recognition using Token Regular Expressions and Bidirectional Inference](https://arxiv.org/abs/1610.05652), 
   Phuong Le-Hong, Proceedings of Vietnamese Speech and Language Processing (VLSP), Hanoi, Vietnam, 2016.

This tool is implemented in the Scala programming language. It utilizes Apache Spark as its core
platform. [Apache Spark](http://spark.apache.org/) is a fast and general engine for large scale data processing. 
Therefore, Vitk.NER is a fast cluster computing toolkit.


## Setup and Compilation ##

* Prerequisites: A Java Development Kit (JDK), version 8.0 or
  later [JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html).
        Apache Maven version 3.0 or later [Maven](http://maven.apache.org/). Make
  sure that two following commands work perfectly in your shell
  (console window).

        `java -version`
        
        `mvn -version`

* Download a prebuilt version of [Apache Spark](https://spark.apache.org/).
        Vitk uses Spark version 2.2.0. Unpack the compressed file to a directory,
        for example `~/spark` where `~` is your home directory.

* Download Vitk.NER, either a binary archive or its source code. The
  repository URL of the project is [Vitk](https://github.com/phuonglh/vn.vitk.ner.git).
  The source code version is preferable. It is easy to compile and
  package Vitk: go to the top-level directory of Vitk and invoke the
  following command at a shell window:

        `mvn compile package`

  Apache Maven will automatically resolve and download dependency
  libraries required by Vitk. Once the process finishes, you should
  have a binary jar file `ai.vitk.ner-1.0.jar` in the sub-directory
  `target`. 

## Running ##

### Data Files ###

Pre-trained model files used by Vitk are included in the resources directory (`src/main/resources`). 

These folders can contain data specific to a natural language in
use. Each language is specified further by a sub-directory whose name
is an abbreviation of the language name, for example `vi` for
Vietnamese, `en` for English, `ja` for Japanese, etc.

Vitk can run as an application on a stand-alone cluster mode  or on a
real cluster. If it is run on a cluster, it is required that
all machines in the cluster are able to access the same data files,
which are normally located in a shared directory readable by all the
machines, or a Hadoop file system.

### Command Line Arguments ###

Vitk.NER is an Apache Spark application, you run it by submitting the 
main JAR file `ai.vitk.ner-1.0.jar` to Apache Spark. The main class of the
toolkit is `ai.vitk.ner.Tagger` which selects the desired tool by following
arguments provided by the user.  

The arguments of Vitk.NER are as follows:

* `--master <master-url>`: the master URL for the cluster, for example
  `spark://192.168.1.1:7077`. If you do not have a cluster, you can
  ignore this parameter. In this case, Vitk uses the stand-alone
  cluster mode, which is defined by `local[*]`, that is, it uses all
  the CPU cores of your single machine for parallel processing.

* `--language <language>`: the natural language to process, where `language` is an abbreviation 
   of language name which is either `vi` (Vietnamese) or `en` (English). If this 
   argument is not specified, the default language is Vietnamese.
  
* `--verbose`: this parameter does not require argument. If it is used, Vitk
   runs in verbose mode, in which some intermediate information
   will be printed out during the processing. This is useful for debugging.

* `--mode <mode>`: the running mode, either `tag`, `train`, or `eval`, the default mode is `tag`.
   
* `--input <input-file>`: the name of an input file to be used. This
   should be a text file in UTF-8 encoding. If the tagger is in the
   `tag` mode, it will read and tag every lines of this file. If it is
   in the `eval` or `train` mode, it will preprocess the file to get
   pairs of word and tag sequences for use in evaluating or training.
    
* `--output <output-file>`: the name of an output file containing the
   tagging results (in the `tag` mode). Since by default, Vitk uses
   Hadoop file system to save results, the output file is actually a
   directory. It 
   contains text files in JSON format, and will be created by
   Vitk. Note that this directory must not already exist, otherwise an
   error will be thrown because Vitk cannot overwrite an existing
   directory. If this parameter is not specified, the result is
   printed to the console window.
    
* `--dimension <dimension>`: this argument is only required in the `train` mode
  to specify the number of features (or the domain dimension) of the
  resulting Conditional Markov Model. The dimension is a positive integer and depends on
  the size of the data. Normally, the larger the training data is, the
  greater the dimension that should be used. Vitk implements the
  [feature hashing](https://en.wikipedia.org/wiki/Feature_hashing) 
  trick, which is a fast and space-efficient way of vectorizing
  features. As an example, we set this argument as 32,768 when
  training a CMM on about 16,000 tagged sentences of the VLSP NER corpus. The default dimension is 32,768.
  
* `--iteration <n>`: this argument is only used in the `train` mode to
  specify the number of iterations used in the L-BFGS optimization algorithm. The default is 600.

* `--reversed`: this parameter does not require argument. If it is used, the tool trains or tests using
    reversed sentences to produce a backward sequence model instead of the default forward sequence model.

* `memory <mem>`: specify the memory used in Spark executors, for example `4g`, `8g`. If you are training a large corpus, 
 you should consider setting an appropriate value for this parameter for Spark efficiency and avoid out of memory errors.

### Running ###

Suppose that Apache Spark has been installed in `~/spark`, Vitk.NER has
been installed in `~/ai.vitk.ner`. To launch Vitk.NER, open a console, enter the
folder `~/ai.vitk.ner` and invoke an appropriate command. For example:

To tag an input file and write the result to an output file of the same name (with generated suffix `.out`), using
  the default pre-trained model:

`./spark/bin/spark-submit target/ai.vitk.ner-1.0.jar --mode tag --input
  <input-file>` 

Because the default mode of the tagger is `tag`, we can therefore drop the argument 
`--mode tag`in the command above.

There is not any `--master` argument in the command above, therefore, Vitk.NER
runs in the stand-alone cluster mode which uses a single local machine and all CPU cores.


* To evaluate the accuracy on a gold corpus in the resource path `/ner/vi/test.txt`, using the default
   pre-trained CMM:

`./spark/bin/spark-submit target/ai.vitk.ner-1.0.jar --mode eval`

* To train a forward model tagger on a gold corpus in the resource path `/ner/vi/train.txt`:

`./spark/bin/spark-submit target/ai.vitk.ner-1.0.jar --mode train --dimension  4096`

The resulting model will be saved in the resource directory `src/main/resources/ner/vi/mlr`.

* To train a backward model tagger on a gold corpus in the resource path `/ner/vi/train.txt`:

`./spark/bin/spark-submit target/ai.vitk.ner-1.0.jar --mode train --reversed `

The resulting model will be saved in the resource directory `src/main/resources/ner/vi/mlr-reversed`.

The resulting CMM has 32,768 dimensions (the default number of dimensions) and is saved to the
default directory `src/main/resources/ner/vi/mlr-reversed/`.

If you wish to run Vitk on a Spark cluster, all you need to do is to
specify the master URL of the cluster, such as: 

`./spark/bin/spark-submit target/ai.vitk.ner-1.0.jar --mode train --reversed --master spark://192.168.1.1:7077` 


### References ###

If you use Vitk.NER in your research, please credit our work by citing the following publication: 

* [Vietnamese Named Entity Recognition using Token Regular Expressions and Bidirectional Inference](https://arxiv.org/abs/1610.05652), 
 Phuong Le-Hong, Proceedings of Vietnamese Speech and Language Processing (VLSP), Hanoi, Vietnam, 2016.


## Contact

Any bug reports, suggestions and collaborations are welcome. I am reachable at:

*    [LE-HONG Phuong](http://mim.hus.vnu.edu.vn/phuonglh)
*    College of Science, Vietnam National University in Hanoi
