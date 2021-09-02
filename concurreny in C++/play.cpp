#include <iostream>
#include <thread>
using namespace std;
/*
void threadFn(int & value) {
   cout<< "I am inside a thread function"<<endl;
   cout<< "Value => "<<value++<<endl;
}
*/
int main() {
   int localvalue = 100;
   thread t1 {[&]{
      cout<< "I am inside a thread function"<<endl;
      cout<< "Value => "<<localvalue++<<endl;
   }};
   t1.join();
   cout<<"Value in the Main Thread => "<<ref(localvalue)<<endl;
   return 0;
}
// main() is where program execution begins.
// int main() {
//    cout << "Hello World"; // prints Hello World
//    return 0;
// }