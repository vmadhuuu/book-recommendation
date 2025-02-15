import React, { useState } from 'react';
import { Search } from 'lucide-react';
import './index.css';

const BookRecommendationApp = () => {
  const [bookTitle, setBookTitle] = useState('');
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!bookTitle.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/recommend/${encodeURIComponent(bookTitle)}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setRecommendations(null);
      } else {
        setRecommendations(data);
        setError(null);
      }
    } catch (err) {
      setError('Failed to fetch recommendations. Please try again.');
      setRecommendations(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h1 className="text-2xl font-bold text-gray-800 mb-6">Book Recommendation System</h1>
          
          {/* Search Input */}
          <div className="flex gap-2 mb-6">
            <div className="relative flex-1">
              <input
                type="text"
                value={bookTitle}
                onChange={(e) => setBookTitle(e.target.value)}
                placeholder="Enter a book title..."
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={loading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 flex items-center gap-2"
            >
              <Search size={20} />
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {/* Results */}
          {recommendations && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-800">
                Recommendations for "{recommendations.book_title}"
              </h2>
              <ul className="space-y-2">
                {recommendations.recommended_books.map((book, index) => (
                  <li
                    key={index}
                    className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    {book}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BookRecommendationApp;