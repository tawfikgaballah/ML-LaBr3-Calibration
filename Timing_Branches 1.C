#define Timing_Branches_cxx
// The class definition in Timing_Branches.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.


// The following methods are defined in this file:
//    Begin():        called every time a loop on the tree starts,
//                    a convenient place to create your histograms.
//    SlaveBegin():   called after Begin(), when on PROOF called only on the
//                    slave servers.
//    Process():      called for each event, in this function you decide what
//                    to read and fill your histograms.
//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//                    called only on the slave servers.
//    Terminate():    called at the end of the loop on the tree,
//                    a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// root> T->Process("Timing_Branches.C")
// root> T->Process("Timing_Branches.C","some options")
// root> T->Process("Timing_Branches.C+")
//


#include "Timing_Branches.h"
#include <TH2.h>
#include <TStyle.h>
#include "TProofServ.h"
#include "TProof.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include<TGraph2D.h>
#include "TObject.h"
#include "TVector.h"
#include <TSelector.h>
void Timing_Branches::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   
}

void Timing_Branches::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

  TString option = GetOption();
  TOutput = new TTree("TOutput","TOutput");
  
  // // Add branches to the output tree
  TOutput->Branch("tdiff", &tdiff, "tdiff/D");
 
  TOutput->Branch("index_i", &index_i, "index_i/I");
  TOutput->Branch("Ei", &Ei, "Ei/D");
  TOutput->Branch("index_j", &index_j, "index_j/I");
  TOutput->Branch("Ej", &Ej, "Ej/D");
   TOutput->Branch("tdiff_aligned", &tdiff_aligned, "tdiff_aligned/D");
   // TOutput->Branch("t_Dynode",&t_Dynode,"t_Dynode/D");
  GetOutputList()->Add(TOutput);
  Time_difference= new TH1D("Time_difference","Time_difference",800,-40,40);
   GetOutputList()->Add(Time_difference);
 Time_difference_aligned= new TH1D("Time_difference_aligned","Time_difference_aligned",800,-40,40);
   GetOutputList()->Add(Time_difference_aligned);
     TOutput->SetNotify(0);
 }

Bool_t Timing_Branches::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // When processing keyed objects with PROOF, the object is already loaded
   // and is available via the fObject pointer.
   //
   // This function should contain the \"body\" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.
  
   fReader.SetLocalEntry(entry);



  // const Long64_t totalEntries = fReader.GetEntries();
  //  const int progressBarWidth = 40;

  //  // Calculate progress percentage and number of '#' characters for the progress bar
  //  int progress = static_cast<int>((entry + 1) * progressBarWidth / static_cast<double>(totalEntries));
  //  int remaining = progressBarWidth - progress;

  //  // Print progress bar
  //  std::cout << "[";
  //  for (int i = 0; i < progress; ++i) {
  //     std::cout << "#";
  //  }
  //  for (int i = 0; i < remaining; ++i) {
  //     std::cout << " ";
  //  }
  //  std::cout << "] " << std::fixed << std::setprecision(1) << (entry + 1) * 100.0 / totalEntries << "%\r";
  //  std::cout.flush();


   // for(int i=0; i<15; i++){
   //   for(int j=i+1; j<15; j++){
   //     double ecal_j= p[j][0]+p[j][1]*(labr3_energy[j]);
   //     double ecal_i= p[i][0]+p[i][1]*(labr3_energy[i]);
   //     if ( labr3_cfdfailbit[i] !=1   && ecal_i>10  && ecal_i<2000 && labr3_time[i]>10 && *pspmt_dycfdfailbit!=1/* &&(i!=0||i!=5||i!=8||i!=11)*/&& *pspmt_dytime>0){
   // 	 if ( labr3_cfdfailbit[j] !=1 && ecal_j>1281  && ecal_j<1382 && /* labr3_ecal[i]>10  && labr3_ecal[i]<450 &&*/ labr3_time[j]>10 && *pspmt_dycfdfailbit!=1/*&&(j!=0||j!=5||j!=8||j!=11)*/){
   // 	   Double_t tdiff_labr3_3D = labr3_time[i]- *pspmt_dytime;
   // 	   if(tdiff_labr3_3D>-100 &&tdiff_labr3_3D<100){
   // 	     Ei = ecal_i;
   // 	     index_i=i;
   // 	     Ej=ecal_j;
   // 	     index_j=15;
   // 	     tdiff= tdiff_labr3_3D;
   // 	     if((*pspmt_dyenergy/10)>50){
   // 	       Dynode=*pspmt_dyenergy/10;
   // 	     }	     
   // 	     TOutput->Fill();
   // 	   }
   // 	 }
   //     }
   //   }
   // }
   
  
   // Double_t T[15][15]{};
   // Double_t EI[15][15]{};
   // Double_t ind_i[15][15]{};
   // Double_t EJ[15][15]{};
   // Double_t ind_j[15][15]{};
       
   //     for(int i=0; i<15; i++){
   // 	 for(int j=i+1; j<15; j++){
   // 	   double ecal_i= (labr3_ecal[i]);
   // 	   double ecal_j= (labr3_ecal[j]);
   // 	   if ( labr3_cfdfailbit[i] !=1   && ecal_i>0  && ecal_i<450 && labr3_time[i]>10 && *pspmt_dycfdfailbit!=1/* &&(i!=0||i!=5||i!=8||i!=11)*/&& *pspmt_dytime>0){
   // 	     if ( labr3_cfdfailbit[j] !=1 && ecal_j>0 && ecal_j<450 && /* labr3_ecal[i]>10  && labr3_ecal[i]<450 &&*/ labr3_time[j]>10 && *pspmt_dycfdfailbit!=1&&(j!=i)){
	  
  	  
   // 	       Double_t tdiff_labr3_3D = labr3_time[i]- labr3_time[j];
   // 	       if(tdiff_labr3_3D>-80 &&tdiff_labr3_3D<80){
   // 		 // fOutputFile << labr3_ecal[i] << "  " << i << "  " << labr3_ecal[j] << "  " << j << "  " << tdiff_labr3_3D  << "\n";
   // 		 // fOutputFile << std::left << std::setw(column1Width) << Ei;
   // 		 // fOutputFile << std::left << std::setw(column2Width) << std::fixed << std::setprecision(2) << i;
   // 		 // fOutputFile << std::left << std::setw(column3Width) << std::fixed << std::setprecision(1) << Ej;
   // 		 // fOutputFile << std::left << std::setw(column4Width) << j;
   // 		 // fOutputFile << std::left << std::setw(column5Width) << std::boolalpha << tdiff_labr3_3D;
   // 		 // fOutputFile << std::endl;
   // 		 // T[i][j]= tdiff_labr3_3D;
   // 		 EI[i][j] = ecal_i;
   // 		 ind_i[i][j]=i;
   // 		 EJ[i][j] = ecal_j;
   // 		 ind_j[i][j]=j;
   // 		 T[i][j]= tdiff_labr3_3D;
   // 		 // if(i!=0 && j!=1){
   // 		 //  tdiff_aligned=T[i][j]-T[0][1];
   // 		 // }
   // 		  TOutput->Fill();
		 
   // 	       }
   // 	     }
   // 	   }
   // 	 }
   //     }


   //     Double_t T_01 = T[0][1];
   //     // Loop through the original time differences and align them
   //     for (int k = 0; k < 15; k++) {
   // 	 for (int l = k+1; l < 15; l++) {
   // 	   if(l!=k){
   // 	     Ei=(double) EI[k][l];
   // 	     index_i=(double) ind_i[k][l];
   // 	     Ej=(double) EJ[k][l] ;
   // 	     index_j= (double) ind_j[k][l];
   // 	       tdiff= (double) T[k][l];
   // 	     // if(k!=0 && l!=1){
   // 	     //   tdiff_aligned= T[k][l]-T_01;
   // 	     // }
   // 	      TOutput->Fill();
   // 	   }
   // 	 }
   //     }
// const int maxEntries = 15 * 15;

// Double_t EcalI[maxEntries];
// Double_t EcalJ[maxEntries];
// Double_t TDiffLabr3_3D[maxEntries];
// Double_t IndexI[maxEntries];
// Double_t IndexJ[maxEntries];

// int entryCount = 0;

   for (int i = 0; i < 15; i++) {
     for (int j = i + 1; j < 15; j++) {
       // double ecal_i = labr3_ecal[i];
       // double ecal_j = labr3_ecal[j];
       if (labr3_cfdfailbit[i] != 1 && labr3_ecal[i] > 0 && labr3_ecal[i] < 1400 && labr3_time[i] > 10 && *pspmt_dycfdfailbit != 1 &&
	   *pspmt_dytime > 0) {
	 if (labr3_cfdfailbit[j] != 1 && labr3_ecal[j] > 0 && labr3_ecal[j] < 1400 && labr3_time[j] > 10 &&
	     *pspmt_dycfdfailbit != 1 && (j != i)) {
	   Double_t tdiff_labr3_3D = labr3_time[i] - labr3_time[j];
	   if (tdiff_labr3_3D > -100 && tdiff_labr3_3D < 100) {
	     Ei = labr3_ecal[i];
	     index_i = i;
	     Ej = labr3_ecal[j];
	     index_j=j;
	     tdiff = tdiff_labr3_3D;
	     if(i==0//||(IndexI[k]==1 && IndexJ[k]==0)
		){
	       tdiff_aligned = tdiff_labr3_3D;
	     }
	     //  if(IndexI[k]!=0 && IndexJ[k]!=8){
      
	     // }
	     else{
	       tdiff_aligned = 0;
	     }
	     TOutput->Fill();
	     // Time_difference->Fill(tdiff);
	     // Time_difference_aligned->Fill(tdiff_aligned);
	   }
	 }
       }
     }
   }

//  Double_t T_01{}; // Initialize to a suitable default value

// // Loop through the original time differences and align them
//  for (int k = 0; k < entryCount; k++) {
//    // for (int l = k + 1; l < entryCount; l++) {
//    //if (l != k) {
//        Ei = EcalI[k];
//        index_i = IndexI[k];
//        Ej = EcalJ[k];
//        index_j = IndexJ[k];
//        tdiff = TDiffLabr3_3D[k];
//        if(IndexI[k]==0//||(IndexI[k]==1 && IndexJ[k]==0)
// 	  ){
// 	 tdiff_aligned = TDiffLabr3_3D[k];
//        }
//        //  if(IndexI[k]!=0 && IndexJ[k]!=8){
      
//        // }
//        else{
//        	 tdiff_aligned = 0;
//         }
//        TOutput->Fill();
//        Time_difference->Fill(TDiffLabr3_3D[k]);
//        Time_difference_aligned->Fill(tdiff_aligned);
//        // }
//      // }
//  }

    
   // atime->Fill(pspmt_atime[1]);



   // std::cout<<"Working"<<'\n';



   
   return kTRUE;
 }

void Timing_Branches::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

}

void Timing_Branches::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

  TFile* fOutputFile=new TFile("/data/e16032/tg1250/ML_From_HPCC/output_Co_2004_no_gates_alignining_method_Trial2.root","RECREATE");
//    OutputObject("Time_difference");
// OutputObject("Time_difference_aligned");
    OutputObject("TOutput");
  // OutputObject("LaBr3_Ecal");
     TOutput->SetNotify(TOutput);
 // TOutput->SetDirectory(fOutputFile);
  TOutput->Write();
 // fOutputFile->Close();
 


  
}
